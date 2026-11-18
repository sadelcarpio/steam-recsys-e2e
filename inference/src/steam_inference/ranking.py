"""Stage 2b (spec 9): the serving ranker re-orders each user's retrieval candidates.

The ranker serves only as part of the serving pair (`evaluation/champion/pair.json`): it runs
when the pair names the retriever this run serves and `rankers/champion/` is that pair's ranker
with the code's feature schema; otherwise the run keeps retrieval order (and says why). Its
features come from training's `steam_training.ranker.features` (one definition): game
attributes of the current catalog + `game_details` (price, release date), popularity from the
`game_popularity_weekly` mart as of this run (the weeks before the current week), and each
user's last liked games.

`RANKER_MODE`: `shadow` computes the ranker order and compares it with retrieval order, but the
run writes retrieval order; `on` writes the ranker's top `RANKER_TOP_K`. A failure on a chunk of
users keeps their retrieval order (counted), never fails the run.

`online_ranker` packs the same ranker for the serving Lambda (`ranker.npz`, published with the
online catalog): its trees and the game side of the features for the published catalog rows,
popularity as of this run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from pyiceberg.exceptions import NoSuchTableError
from steam_training.artifacts import ArtifactStore
from steam_training.config import CHAMPION
from steam_training.data import Catalog, Ragged, TableSource
from steam_training.pairs import load_ranker
from steam_training.ranker.data import (
    POPULARITY_MART,
    epoch_seconds,
    game_details,
    popularity_from_mart,
)
from steam_training.ranker.export import encode_ranker, ranker_arrays
from steam_training.ranker.features import (
    FEATURES,
    MAX_RECENT_WEEKS,
    GameTable,
    PopularityTable,
    compute_features,
    week_of,
)
from steam_training.ranker.model import FitResult

from steam_inference.features import InferenceData
from steam_inference.online import RankerPayload
from steam_inference.retrieval import NO_ROW, Candidates

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingRanker:
    ranker_id: str
    fit: FitResult
    history_length: int
    games: GameTable
    popularity: PopularityTable
    snapshots: dict[str, int | None] = field(default_factory=dict)


@dataclass
class RankerStats:
    ranked_users: int = 0
    failures: int = 0
    # mean share of the retrieval top RANKER_TOP_K also in the ranker's top RANKER_TOP_K
    overlap: float | None = None
    seconds: float = 0.0
    # per user: True when the ranker ordered its candidates (False: a failed chunk)
    ranked: np.ndarray | None = None


def load_serving_ranker(
    source: TableSource,
    store: ArtifactStore,
    catalog: Catalog,
    retriever_id: str,
    as_of: datetime,
    history_length: int | None = None,
) -> tuple[ServingRanker | None, str]:
    """(the serving ranker, why) or (None, why not), with the game side of its features for
    `catalog` (the current games). Popularity covers the weeks before `as_of`'s week (what
    training reads at t = `as_of`). `history_length`: the users' history this run has (warns
    when shorter than the ranker's)."""
    pair = store.read_pair()
    if pair is None or pair.ranker_id is None:
        return None, "no ranker in the serving pair"
    if pair.retriever_id != retriever_id:
        return None, (
            f"the serving pair's retriever is {pair.retriever_id}, this run serves "
            f"{retriever_id}: ranker {pair.ranker_id} was not evaluated with it"
        )
    try:
        fit, metadata = load_ranker(store, CHAMPION)
    except (FileNotFoundError, ValueError) as err:
        return None, f"champion ranker unusable: {err}"
    if metadata.ranker_id != pair.ranker_id:
        return None, (
            f"rankers/champion/ holds {metadata.ranker_id}, the serving pair names {pair.ranker_id}"
        )
    try:
        snapshots = {POPULARITY_MART: source.snapshot_id(POPULARITY_MART)}
        snapshots |= {t: source.snapshot_id(t) for t in ("lkp_games", "game_details")}
    except NoSuchTableError as err:
        return None, f"mart missing for the ranker features: {err}"
    num_games = len(catalog.row_of)
    items = catalog.items
    price, release_day = game_details(source, snapshots, num_games)
    games = GameTable.from_catalog(
        catalog,
        num_games,
        _size(items.tags),
        _size(items.genres),
        _size(items.categories),
        price=price,
        release_day=release_day,
    )
    week = int(week_of(epoch_seconds(as_of)))
    popularity = popularity_from_mart(
        source, snapshots[POPULARITY_MART], num_games, week - MAX_RECENT_WEEKS - 1, week
    )
    history = history_length if history_length is not None else metadata.history_length
    if history < metadata.history_length:
        log.warning(
            "the retriever's history (%d games) is shorter than the ranker's (%d): its "
            "user-side features read %d games",
            history,
            metadata.history_length,
            history,
        )
    ranker = ServingRanker(
        ranker_id=metadata.ranker_id,
        fit=fit,
        history_length=metadata.history_length,
        games=games,
        popularity=popularity,
        snapshots=snapshots,
    )
    return ranker, f"ranker {metadata.ranker_id} (pair with {retriever_id})"


def _size(values: Ragged) -> int:
    return int(values.values.max()) + 1 if len(values.values) else 0


def apply_ranker(
    ranker: ServingRanker,
    data: InferenceData,
    candidates: Candidates,
    *,
    as_of: datetime,
    top_k: int,
    chunk: int,
) -> tuple[Candidates, RankerStats]:
    """Candidates re-ordered by the ranker (all of them; the caller cuts the list) and stats.
    Features are as of `as_of` (popularity: the weeks before its week)."""
    started = time.monotonic()
    rows, scores = candidates.rows, candidates.scores
    n, k = rows.shape
    valid = rows != NO_ROW
    game = np.where(valid, data.games.catalog.items.game_idx[np.maximum(rows, 0)], -1)
    retrieval_scores = np.where(valid, scores, -np.inf).astype(np.float32)
    history = data.users.history[:, : ranker.history_length]
    seconds = np.full(n, epoch_seconds(as_of), dtype=np.int64)
    order = np.broadcast_to(np.arange(k), (n, k)).copy()
    ranked = np.zeros(n, dtype=bool)
    stats = RankerStats(ranked=ranked)
    for start in range(0, n, chunk):
        part = slice(start, min(start + chunk, n))
        try:
            x = compute_features(
                history[part], game[part], retrieval_scores[part], seconds[part],
                ranker.games, ranker.popularity,
            )  # fmt: skip
            score = ranker.fit.predict_features(x.reshape(-1, len(FEATURES))).reshape(x.shape[:2])
            score[~valid[part]] = -np.inf
            order[part] = np.argsort(-score, axis=1, kind="stable")
            stats.ranked_users += part.stop - part.start
            ranked[part] = True
        except Exception:  # one chunk keeps retrieval order, the run goes on
            stats.failures += part.stop - part.start
            log.exception("ranker failed on users %d-%d: retrieval order kept", start, part.stop)
    ranked_rows = np.take_along_axis(rows, order, axis=1)
    ranked = Candidates(
        rows=ranked_rows,
        scores=np.take_along_axis(scores, order, axis=1),
        item_embeddings=candidates.item_embeddings,
    )
    stats.overlap = top_overlap(rows, ranked_rows, top_k)
    stats.seconds = round(time.monotonic() - started, 1)
    log.info(
        "ranker %s: %d users ranked, %d kept retrieval order, overlap@%d %.3f, %.0fs",
        ranker.ranker_id,
        stats.ranked_users,
        stats.failures,
        top_k,
        stats.overlap or 0.0,
        stats.seconds,
    )
    return ranked, stats


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float | None:
    """Mean over users of |top k of a ∩ top k of b| / (games in a's top k)."""
    a, b = a[:, :k], b[:, :k]
    if not len(a):
        return None
    width = int(max(a.max(initial=0), b.max(initial=0))) + 2
    owner = np.arange(len(a))[:, None] * width
    shared = np.isin(owner + (a + 1), owner + (b + 1)) & (a != NO_ROW)
    size = np.maximum((a != NO_ROW).sum(axis=1), 1)
    return float((shared.sum(axis=1) / size).mean())


def cut(candidates: Candidates, k: int) -> Candidates:
    return Candidates(candidates.rows[:, :k], candidates.scores[:, :k], candidates.item_embeddings)


def online_ranker(
    ranker: ServingRanker,
    game_idx: np.ndarray,
    *,
    retriever_id: str,
    candidates: int,
    mode: str,
    as_of: datetime,
) -> RankerPayload:
    """`ranker.npz` for the published catalog's rows (`game_idx` of each row, in order)."""
    arrays = ranker_arrays(
        ranker.fit,
        ranker_id=ranker.ranker_id,
        model_id=retriever_id,
        history_length=ranker.history_length,
        candidates=candidates,
        games=ranker.games,
        popularity=ranker.popularity,
        game_idx=game_idx,
        as_of_seconds=epoch_seconds(as_of),
    )
    return RankerPayload(encode_ranker(arrays), ranker.ranker_id, mode)
