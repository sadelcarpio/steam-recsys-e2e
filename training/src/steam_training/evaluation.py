"""Stage 4, evaluation: recall@K of sampled positive validation rows against the full catalog.

For every positive validation review, the user's history (as of before the review) scores every
game of the catalog as of the split cutoff. Games already in the history are excluded, and the row
is a hit at K when the reviewed game ranks in the top K. There is one relevant item per row, so
recall@K is the hit rate. Rows are reported as warm (non-empty history, the inference
population), cold (empty history) and all. Validation rows are a uniform sample (at most
`EVAL_MAX_ROWS`), so every segment is unbiased.

The scoring loops (evaluation and hard-negative mining) are pure torch and run on `device`;
numpy only appears at the entry points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from steam_training.batching import to_item_batch
from steam_training.contracts import RecallMetrics, SegmentRecall
from steam_training.data import Catalog, EvalRows
from steam_training.model import TwoTowerModel

NOT_FOUND = torch.iinfo(torch.int64).max
# history chunk [b, 5] (on device) -> fresh (writable) scores [b, catalog]; callers mask them
# in place
ScoreFn = Callable[[torch.Tensor], torch.Tensor]
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

    @torch.no_grad()
    def score(history: torch.Tensor) -> torch.Tensor:
        return model.score(model.user_tower(history), item_embeddings)

    return score


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

    def score(history: torch.Tensor) -> torch.Tensor:
        return item_scores.repeat(len(history), 1)

    return score


def mask_games(scores: torch.Tensor, games: torch.Tensor, index: CatalogIndex) -> torch.Tensor:
    """Set scores[i, row of games[i, j]] to -inf for every game in the catalog, in place.
    `games` is [b, m] (e.g. histories); padding and unknown games are ignored."""
    rows = index.rows(games)
    user, position = torch.nonzero(rows >= 0, as_tuple=True)
    scores[user, rows[user, position]] = float("-inf")
    return scores


def target_ranks(
    score_fn: ScoreFn,
    history: torch.Tensor,
    target: torch.Tensor,
    index: CatalogIndex,
    max_k: int,
    batch_size: int,
) -> torch.Tensor:
    """0-based rank of each row's target in its top `max_k`, history excluded (NOT_FOUND when
    outside). Chunks move to the index's device; ranks come back on the CPU."""
    device = index.game_idx.device
    ranks = torch.full((len(target),), NOT_FOUND, dtype=torch.int64)
    k = min(max_k, len(index))
    for start in range(0, len(target), batch_size):
        chunk = slice(start, start + batch_size)
        chunk_history = history[chunk].to(device)
        scores = mask_games(score_fn(chunk_history), chunk_history, index)
        retrieved = index.game_idx[torch.topk(scores, k, dim=1).indices]
        hit = retrieved == target[chunk, None].to(device)
        found = hit.any(dim=1)
        chunk_ranks = torch.full((len(hit),), NOT_FOUND, dtype=torch.int64, device=device)
        chunk_ranks[found] = hit[found].int().argmax(dim=1)
        ranks[chunk] = chunk_ranks.cpu()
    return ranks


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
) -> RecallMetrics:
    ranks = target_ranks(
        score_fn,
        torch.from_numpy(rows.history),
        torch.from_numpy(rows.target),
        CatalogIndex.of(catalog, device),
        max(ks),
        batch_size,
    )
    return recall_from_ranks(ranks, torch.from_numpy(rows.is_warm), ks)


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
