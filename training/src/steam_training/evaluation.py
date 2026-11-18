"""Stage 4, evaluation: recall@K of sampled positive validation rows against the full catalog.

For every positive validation review, the user's history (as of before the review) scores every
game of the catalog as of the split cutoff. Games the user already reviewed before that review
(any polarity, `EvalRows.reviewed`; the 5-game history when unknown) are excluded, and the row
is a hit at K when the reviewed game ranks in the top K. There is one relevant item per row, so
recall@K is the hit rate. Rows are reported as warm (non-empty history, the inference
population), cold (empty history) and all. Validation rows are a uniform sample (at most
`EVAL_MAX_ROWS`), so every segment is unbiased.

A6 (item recency): inference re-embeds the catalog on every run, so recency features as of the
cutoff would be months stale for most validation rows. A model that reads them scores each
row against the catalog as of the row's ISO week (Monday UTC), from the reviews of the weeks
before it (`game_popularity_weekly`), as a weekly inference run would. Other models (and the
popularity baseline) score as before.

The scoring loops (evaluation and hard-negative mining) are pure torch and run on `device`;
numpy only appears at the entry points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from steam_training.batching import to_item_batch
from steam_training.contracts import Concentration, RecallMetrics, SegmentRecall
from steam_training.data import (
    DAY_SECONDS,
    Catalog,
    EvalRows,
    Ragged,
    week_of_day,
    week_start_day,
)
from steam_training.model import TwoTowerModel

NOT_FOUND = torch.iinfo(torch.int64).max
# history chunk [b, 5] (on device) [, A7: language ids [b]] -> fresh (writable) scores
# [b, catalog]; callers mask them in place
ScoreFn = Callable[..., torch.Tensor]
# A6: the score function of a model over another catalog (recency as of another day)
Rescore = Callable[[Catalog], ScoreFn]
# the score function as of an ISO week (`weekly_scores`)
WeeklyScores = Callable[[int], ScoreFn]
CPU = torch.device("cpu")


@dataclass(frozen=True)
class CatalogIndex:
    """Tensor view of a `Catalog` for the scoring loops."""

    game_idx: torch.Tensor  # int64 [catalog]: game of each catalog row
    row_of: torch.Tensor  # int64 [games]: catalog row of each game_idx, -1 when absent

    def __len__(self) -> int:
        return len(self.game_idx)

    @classmethod
    def of(cls, catalog: Catalog, device: torch.device = CPU) -> CatalogIndex:
        # copies: game_idx may be a read-only zero-copy view of the Arrow table
        return cls(
            torch.tensor(catalog.items.game_idx, device=device),
            torch.tensor(catalog.row_of, device=device),
        )

    def rows(self, game_idx: torch.Tensor) -> torch.Tensor:
        """Catalog rows of `game_idx` (-1 for games outside the catalog / vocabulary)."""
        inside = (game_idx >= 0) & (game_idx < len(self.row_of))
        rows = self.row_of[game_idx.clamp(0, len(self.row_of) - 1)]
        return torch.where(inside, rows, torch.full_like(rows, -1))


@torch.no_grad()
def embed_catalog(
    model: TwoTowerModel, catalog: Catalog, batch_size: int, device: torch.device = CPU
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(catalog), batch_size):
        items = catalog.items.take(np.arange(start, min(start + batch_size, len(catalog))))
        chunks.append(model.item_tower(to_item_batch(items).to(device)))
    return torch.cat(chunks)


def model_scores(
    model: TwoTowerModel, catalog: Catalog, batch_size: int, device: torch.device = CPU
) -> ScoreFn:
    model.eval()
    item_embeddings = embed_catalog(model, catalog, batch_size, device)

    length = model.user_tower.history_length

    @torch.no_grad()
    def score(history: torch.Tensor, language: torch.Tensor | None = None) -> torch.Tensor:
        # histories are most recent first: a model with a shorter history reads the prefix
        # (promotion scores models of different lengths on the same rows)
        return model.score(model.user_tower(history[:, :length], language), item_embeddings)

    return score


def recency_rescore(
    model: TwoTowerModel, batch_size: int, device: torch.device = CPU
) -> Rescore | None:
    """For a model that reads item recency (A6): its scores over a refreshed catalog."""
    if not model.config.item_recency:
        return None
    return lambda catalog: model_scores(model, catalog, batch_size, device)


def weekly_scores(
    model: TwoTowerModel, catalog: Catalog, batch_size: int, device: torch.device = CPU
) -> WeeklyScores:
    """A model's score function as of an ISO week: an A6 model scores against the catalog with
    recency as of that week (weeks up to the catalog's own keep its recency, as `rank_rows`);
    any other model has one score function. The last week's embeddings are cached (callers go
    through the weeks in order)."""
    base = model_scores(model, catalog, batch_size, device)
    if not model.config.item_recency or catalog.as_of_day is None:
        return lambda week: base
    start = int(week_of_day(catalog.as_of_day))
    cache: dict[int, ScoreFn] = {}

    def at(week: int) -> ScoreFn:
        week = int(week)
        if week <= start:
            return base
        if week not in cache:
            cache.clear()
            cache[week] = model_scores(
                model, catalog.at_day(int(week_start_day(week))), batch_size, device
            )
        return cache[week]

    return at


def recent_popularity_scores(catalog: Catalog, device: torch.device = CPU) -> ScoreFn:
    """A6 baseline: every user gets the games with the most reviews in the 4 weeks before the
    catalog's ISO week (13 weeks break ties). With `recent_popularity_rescore` the catalog is
    refreshed weekly, like an A6 model's: "what's trending this week", no model."""
    recency = catalog.items.recency
    if recency is None:
        raise ValueError("recent popularity needs a catalog with recency features (A6)")
    item_scores = torch.tensor(recency[:, 0] + 1e-3 * recency[:, 1], device=device)

    def score(history: torch.Tensor, language: torch.Tensor | None = None) -> torch.Tensor:
        return item_scores.repeat(len(history), 1)

    return score


def recent_popularity_rescore(device: torch.device = CPU) -> Rescore:
    return lambda catalog: recent_popularity_scores(catalog, device)


def popularity_scores(
    positive_counts: np.ndarray, catalog: Catalog, device: torch.device = CPU
) -> ScoreFn:
    """Baseline: every user gets the most reviewed-positive games of the training split
    (`positive_counts[game_idx]`)."""
    counts = torch.tensor(positive_counts, device=device)
    games = CatalogIndex.of(catalog, device).game_idx
    inside = games < len(counts)
    item_scores = torch.where(
        inside, counts[games.clamp(max=len(counts) - 1)], torch.zeros_like(games)
    ).float()

    def score(history: torch.Tensor, language: torch.Tensor | None = None) -> torch.Tensor:
        return item_scores.repeat(len(history), 1)

    return score


def mask_games(scores: torch.Tensor, games: torch.Tensor, index: CatalogIndex) -> torch.Tensor:
    """Set scores[i, row of games[i, j]] to -inf for every game in the catalog, in place.
    `games` is [b, m] (e.g. histories); padding and unknown games are ignored."""
    rows = index.rows(games)
    user, position = torch.nonzero(rows >= 0, as_tuple=True)
    scores[user, rows[user, position]] = float("-inf")
    return scores


@dataclass(frozen=True)
class Exposure:
    """What a scoring pass retrieved, per catalog row (on the index's device)."""

    best_rank: torch.Tensor  # int64 [catalog]: best 0-based rank over all rows, NOT_FOUND if never
    count: torch.Tensor  # int64 [catalog]: rows that retrieved it in their top max_k

    @classmethod
    def empty(cls, index: CatalogIndex) -> Exposure:
        device = index.game_idx.device
        return cls(
            torch.full((len(index),), NOT_FOUND, dtype=torch.int64, device=device),
            torch.zeros(len(index), dtype=torch.int64, device=device),
        )

    def add(self, top_rows: torch.Tensor) -> None:
        """`top_rows` [b, k]: catalog rows of each query's top k, best first."""
        flat = top_rows.reshape(-1)
        position = torch.arange(top_rows.shape[1], device=flat.device).repeat(len(top_rows))
        self.best_rank.scatter_reduce_(0, flat, position, reduce="amin")
        self.count.index_add_(0, flat, torch.ones_like(flat))


def _mask_ragged(
    scores: torch.Tensor, games: Ragged, start: int, stop: int, index: CatalogIndex
) -> None:
    """-inf for every game of rows [start, stop) of `games` (scores row 0 = row `start`)."""
    lo, hi = int(games.offsets[start]), int(games.offsets[stop])
    if hi == lo:
        return
    device = scores.device
    lengths = torch.from_numpy(np.diff(games.offsets[start : stop + 1])).to(device)
    owner = torch.repeat_interleave(torch.arange(stop - start, device=device), lengths)
    rows = index.rows(torch.from_numpy(games.values[lo:hi]).to(device))
    inside = rows >= 0
    scores[owner[inside], rows[inside]] = float("-inf")


def target_ranks(
    score_fn: ScoreFn,
    history: torch.Tensor,
    target: torch.Tensor,
    index: CatalogIndex,
    max_k: int,
    batch_size: int,
    exposure: Exposure | None = None,
    exclude: Ragged | None = None,
    language: torch.Tensor | None = None,
) -> torch.Tensor:
    """0-based rank of each row's target in its top `max_k`, history excluded (NOT_FOUND when
    outside). Chunks move to the index's device; ranks come back on the CPU. `exposure` given:
    also accumulates what every row retrieved. `exclude` given: those games (per row, e.g.
    everything the user already reviewed) are excluded too."""
    device = index.game_idx.device
    ranks = torch.full((len(target),), NOT_FOUND, dtype=torch.int64)
    k = min(max_k, len(index))
    for start in range(0, len(target), batch_size):
        chunk = slice(start, start + batch_size)
        chunk_history = history[chunk].to(device)
        if language is None:
            raw = score_fn(chunk_history)
        else:  # A7: the user's language
            raw = score_fn(chunk_history, language[chunk].to(device))
        scores = mask_games(raw, chunk_history, index)
        if exclude is not None:
            _mask_ragged(scores, exclude, start, start + len(chunk_history), index)
        top_rows = torch.topk(scores, k, dim=1).indices
        if exposure is not None:
            exposure.add(top_rows)
        retrieved = index.game_idx[top_rows]
        hit = retrieved == target[chunk, None].to(device)
        found = hit.any(dim=1)
        chunk_ranks = torch.full((len(hit),), NOT_FOUND, dtype=torch.int64, device=device)
        chunk_ranks[found] = hit[found].int().argmax(dim=1)
        ranks[chunk] = chunk_ranks.cpu()
    return ranks


def top_candidates(
    score_fn: ScoreFn,
    history: np.ndarray,
    index: CatalogIndex,
    k: int,
    batch_size: int,
    exclude: Ragged | None = None,
    language: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's top `k` games (game_idx, best first) and their scores, with the same
    exclusions as `target_ranks` (the history, and `exclude` per row). Excluded games score
    -inf, so a catalog with fewer than `k` eligible games yields -inf tails. `language` (A7):
    each row's language id (None: unknown)."""
    device = index.game_idx.device
    k = min(k, len(index))
    games = np.zeros((len(history), k), dtype=np.int64)
    scores = np.zeros((len(history), k), dtype=np.float32)
    for start in range(0, len(history), batch_size):
        chunk_history = torch.from_numpy(history[start : start + batch_size]).to(device)
        chunk_language = (
            torch.from_numpy(language[start : start + batch_size].astype(np.int64)).to(device)
            if language is not None
            else None
        )
        chunk = mask_games(score_fn(chunk_history, chunk_language), chunk_history, index)
        if exclude is not None:
            _mask_ragged(chunk, exclude, start, start + len(chunk_history), index)
        top = torch.topk(chunk, k, dim=1)
        end = start + len(chunk_history)
        games[start:end] = index.game_idx[top.indices].cpu().numpy()
        scores[start:end] = top.values.float().cpu().numpy()
    return games, scores


def weekly_top_candidates(
    scores_at: WeeklyScores,
    history: np.ndarray,
    seconds: np.ndarray,
    index: CatalogIndex,
    k: int,
    batch_size: int,
    exclude: Ragged | None = None,
    language: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`top_candidates` with each row scored as of its ISO week (`seconds`: its time)."""
    week = week_of_day(np.asarray(seconds, np.int64) // DAY_SECONDS)
    k = min(k, len(index))
    games = np.zeros((len(history), k), dtype=np.int64)
    scores = np.zeros((len(history), k), dtype=np.float32)
    for w in np.unique(week):
        part = np.flatnonzero(week == w)
        games[part], scores[part] = top_candidates(
            scores_at(int(w)),
            history[part],
            index,
            k,
            batch_size,
            exclude=exclude.take(part) if exclude is not None else None,
            language=language[part] if language is not None else None,
        )
    return games, scores


def concentration(
    score_fn: ScoreFn,
    history: np.ndarray,
    index: CatalogIndex,
    seen: np.ndarray,
    k: int,
    batch_size: int,
    exclude: Ragged | None = None,
    language: np.ndarray | None = None,
) -> Concentration:
    """`Concentration` of the top-`k` lists of `history` rows over the catalog `index`
    (`seen`: bool per game_idx the model trained on; beyond it: unseen)."""
    games, scores = top_candidates(
        score_fn, history, index, k, batch_size, exclude=exclude, language=language
    )
    return list_concentration(np.where(np.isfinite(scores), games, -1), seen)


def list_concentration(lists: np.ndarray, seen: np.ndarray) -> Concentration:
    """`Concentration` of ready lists: int64 [users, k] game_idx, best first, -1 = empty slot
    (e.g. a ranker's re-ordered top k)."""
    users, k = lists.shape
    listed = lists[lists >= 0]
    if not users or not len(listed):
        return Concentration(
            k=k, lists=users, reach=0.0, top_game_idx=-1, top10_share=0.0, unseen_share=0.0,
            distinct=0,
        )  # fmt: skip
    ids, counts = np.unique(listed, return_counts=True)  # a game appears once per list
    order = np.argsort(-counts, kind="stable")
    inside = listed < len(seen)
    known = np.zeros(len(listed), bool)
    known[inside] = seen[listed[inside]]
    return Concentration(
        k=k,
        lists=users,
        reach=float(counts[order[0]]) / users,
        top_game_idx=int(ids[order[0]]),
        top10_share=float(counts[order[:10]].sum()) / len(listed),
        unseen_share=float((~known).mean()),
        distinct=len(ids),
    )


def model_concentration(
    model: TwoTowerModel,
    catalog: Catalog,
    history: np.ndarray,
    k: int,
    batch_size: int,
    device: torch.device = CPU,
    exclude: Ragged | None = None,
    language: np.ndarray | None = None,
) -> Concentration:
    """`concentration` of a model over `catalog` (normally `data.current_catalog`)."""
    seen = model.item_tower.game_ids.seen_games.detach().cpu().numpy()
    return concentration(
        model_scores(model, catalog, batch_size, device),
        history,
        CatalogIndex.of(catalog, device),
        seen,
        k,
        batch_size,
        exclude=exclude,
        language=language,
    )


def recall_from_ranks(ranks: torch.Tensor, is_warm: torch.Tensor, ks: list[int]) -> RecallMetrics:
    def segment(mask: torch.Tensor) -> SegmentRecall:
        n = int(mask.sum())
        selected = ranks[mask]
        recall = {k: float((selected < k).float().mean()) if n else 0.0 for k in sorted(ks)}
        return SegmentRecall(rows=n, recall=recall)

    return RecallMetrics(
        warm=segment(is_warm),
        cold=segment(~is_warm),
        all=segment(torch.ones_like(is_warm)),
    )


def evaluate(
    score_fn: ScoreFn,
    rows: EvalRows,
    catalog: Catalog,
    ks: list[int],
    batch_size: int,
    device: torch.device = CPU,
    rescore: Rescore | None = None,
) -> RecallMetrics:
    ranks, _ = rank_rows(score_fn, rows, catalog, max(ks), batch_size, device, rescore)
    return recall_from_ranks(ranks, torch.from_numpy(rows.is_warm), ks)


def rank_rows(
    score_fn: ScoreFn,
    rows: EvalRows,
    catalog: Catalog,
    max_k: int,
    batch_size: int,
    device: torch.device = CPU,
    rescore: Rescore | None = None,
) -> tuple[torch.Tensor, Exposure]:
    """Target ranks of `rows` (CPU) and what the scoring pass retrieved (`Exposure`).
    `rescore` (an A6 model's, `recency_rescore`): each row is scored against the catalog with
    recency as of its ISO week (rows of the catalog's own week keep `score_fn`)."""
    index = CatalogIndex.of(catalog, device)
    exposure = Exposure.empty(index)

    def ranks_of(fn: ScoreFn, part: EvalRows) -> torch.Tensor:
        return target_ranks(
            fn,
            torch.from_numpy(part.history),
            torch.from_numpy(part.target),
            index,
            max_k,
            batch_size,
            exposure,
            part.reviewed,
            torch.from_numpy(part.language.astype(np.int64)) if part.language is not None else None,
        )

    if rescore is None or catalog.as_of_day is None or rows.timestamp is None:
        return ranks_of(score_fn, rows), exposure
    start = int(week_of_day(catalog.as_of_day))
    week = np.maximum(week_of_day(np.asarray(rows.timestamp) // DAY_SECONDS), start)
    ranks = torch.full((len(rows),), NOT_FOUND, dtype=torch.int64)
    for w in np.unique(week):
        part = np.flatnonzero(week == w)
        fn = score_fn if w == start else rescore(catalog.at_day(int(week_start_day(w))))
        ranks[torch.from_numpy(part)] = ranks_of(fn, rows.take(part))
    return ranks, exposure


@torch.no_grad()
def mine_hard_negatives(
    model: TwoTowerModel,
    history: np.ndarray,
    target: np.ndarray,
    catalog: Catalog,
    *,
    skip_top: int,
    pool_size: int,
    generator: torch.Generator,
    batch_size: int,
    device: torch.device = CPU,
) -> np.ndarray:
    """One game per example sampled from its ranks [skip_top, skip_top + pool_size) under the
    current model, excluding the target and the history. `generator` is a CPU generator."""
    score_fn = model_scores(model, catalog, batch_size, device)
    index = CatalogIndex.of(catalog, device)
    history_t, target_t = torch.from_numpy(history), torch.from_numpy(target)
    depth = min(skip_top + pool_size, len(index))
    first = min(skip_top, depth - 1)
    mined = torch.empty(len(target_t), dtype=torch.int64)
    for start in range(0, len(target_t), batch_size):
        chunk = slice(start, start + batch_size)
        chunk_history = history_t[chunk].to(device)
        scores = mask_games(score_fn(chunk_history), chunk_history, index)
        mask_games(scores, target_t[chunk, None].to(device), index)
        top = torch.topk(scores, depth, dim=1).indices
        pick = torch.randint(first, depth, (len(top), 1), generator=generator).to(device)
        mined[chunk] = index.game_idx[top.gather(1, pick).squeeze(1)].cpu()
    return mined.numpy()
