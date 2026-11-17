"""Ranker features (spec 9): one definition for training, batch inference and serving.

For a user at time t with history H (the last `history_length` liked / positively reviewed
games, most recent first, padded with PADDING_ID) and the retriever's candidates (best first,
with their scores), `compute_features` returns a [groups, candidates, len(FEATURES)] matrix.
Every user-side feature is a function of H only (what an online request carries), and no
feature depends on a retriever version: retrieval enters through scale-free positions only.

Game-side inputs come from a `GameTable` (static attributes per game_idx) and a
`PopularityTable` (positive / total reviews per game and week; a feature at t reads only the
weeks strictly before t's week). Pure numpy: no torch, no LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from steam_training.contracts import FIRST_ID
from steam_training.data import Catalog, Ragged

# Bump when a feature's definition (or the list) changes: a ranker only runs with the schema it
# was trained on.
# 2: similarity (tag cosine max / mean), fit (popularity / release / price fit, free share) and
#    user (history size) families removed: no recall effect on the champion's candidates
#    (ablation of ranker local-3acaec2-rank1, 2026-11-17)
FEATURE_SCHEMA_VERSION = 2

FAMILIES: dict[str, list[str]] = {
    "retrieval": ["rank", "score_percentile", "score_gap"],
    "candidate": [
        "log_positives",
        "log_positives_30d",
        "log_positives_90d",
        "positive_ratio",
        "is_free",
        "log_price",
        "age_days",
    ],
    "affinity": [
        "tag_cosine",
        "developer_share",
        "publisher_share",
        "genre_overlap",
        "category_overlap",
    ],
}
FEATURES = [f for features in FAMILIES.values() for f in features]

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
# 30 / 90 days, in whole weeks before t's week
RECENT_WEEKS = {"log_positives_30d": 4, "log_positives_90d": 13}
MAX_RECENT_WEEKS = max(RECENT_WEEKS.values())


def week_of(seconds: np.ndarray | int) -> np.ndarray | int:
    """Week index (seconds since the epoch // one week); popularity tables use the same one."""
    return np.asarray(seconds) // SECONDS_PER_WEEK


@dataclass(frozen=True)
class GameTable:
    """Static game attributes, row = game_idx (rows outside the catalog stay empty / NaN)."""

    tags: np.ndarray  # float32 [games, tags], tag weights, L2-normalized rows (0 = no tags)
    developer: np.ndarray  # int64 [games], first developer id (-1 = none)
    publisher: np.ndarray  # int64 [games], first publisher id (-1 = none)
    genres: np.ndarray  # float32 [games, genres] multi-hot
    categories: np.ndarray  # float32 [games, categories] multi-hot
    is_free: np.ndarray  # float32 [games]
    log_price: np.ndarray  # float32 [games], log1p(price); NaN = unknown
    release_day: np.ndarray  # float32 [games], days since the epoch; NaN = unknown

    @property
    def num_games(self) -> int:
        return len(self.is_free)

    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        num_games: int,
        num_tags: int,
        num_genres: int,
        num_categories: int,
        price: np.ndarray | None = None,
        release_day: np.ndarray | None = None,
    ) -> GameTable:
        """`price` / `release_day`: per game_idx (NaN unknown), e.g. from `game_details`."""
        items = catalog.items
        games = items.game_idx
        tags = np.zeros((num_games, max(num_tags, 1)), dtype=np.float32)
        _fill_dense(tags, games, items.tags, weighted=True)
        tags /= np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1e-12)
        genres = np.zeros((num_games, max(num_genres, 1)), dtype=np.float32)
        _fill_dense(genres, games, items.genres)
        categories = np.zeros((num_games, max(num_categories, 1)), dtype=np.float32)
        _fill_dense(categories, games, items.categories)
        is_free = np.zeros(num_games, dtype=np.float32)
        is_free[games] = items.is_free
        nan = np.full(num_games, np.nan, dtype=np.float32)
        log_price = nan.copy() if price is None else np.log1p(price).astype(np.float32)
        return cls(
            tags=tags,
            developer=_first(games, items.developers, num_games),
            publisher=_first(games, items.publishers, num_games),
            genres=genres,
            categories=categories,
            is_free=is_free,
            log_price=log_price,
            release_day=nan.copy() if release_day is None else release_day.astype(np.float32),
        )


def _fill_dense(out: np.ndarray, games: np.ndarray, lists: Ragged, weighted: bool = False):
    lengths = np.diff(lists.offsets)
    rows = np.repeat(games, lengths)
    cols = lists.values
    inside = (cols >= 0) & (cols < out.shape[1])
    if weighted and lists.weights is not None:
        values = lists.weights
    else:
        values = np.ones(len(cols), dtype=np.float32)
    np.add.at(out, (rows[inside], cols[inside]), values[inside])
    if not weighted:
        np.minimum(out, 1, out=out)


def _first(games: np.ndarray, lists: Ragged, num_games: int) -> np.ndarray:
    out = np.full(num_games, -1, dtype=np.int64)
    lengths = np.diff(lists.offsets)
    has = lengths > 0
    out[games[has]] = lists.values[lists.offsets[:-1][has]]
    return out


@dataclass(frozen=True)
class PopularityTable:
    """Cumulative positive / total reviews per game: row j counts the weeks < `first_week` + j
    (row 0 = everything before `first_week`). A lookup for time t reads the row of t's week, so
    t's own week is never counted (the same definition as the `game_popularity_weekly` mart)."""

    first_week: int
    positives: np.ndarray  # int64 [weeks + 1, games]
    reviews: np.ndarray  # int64 [weeks + 1, games]

    @classmethod
    def from_weekly(
        cls,
        game: np.ndarray,
        week: np.ndarray,
        positives: np.ndarray,
        reviews: np.ndarray,
        num_games: int,
        first_week: int,
        last_week: int,
    ) -> PopularityTable:
        """From weekly counts (one row per game and week). Weeks after `last_week` are ignored;
        lookups for later weeks get the last row."""
        weeks = last_week - first_week + 1
        inside = (game >= 0) & (game < num_games) & (week <= last_week)
        game, week = game[inside], week[inside]
        slot = np.clip(week - first_week + 1, 0, weeks)  # weeks before first_week -> row 0
        pos = np.zeros((weeks + 1) * num_games, dtype=np.int64)
        tot = np.zeros((weeks + 1) * num_games, dtype=np.int64)
        np.add.at(pos, slot * num_games + game, positives[inside])
        np.add.at(tot, slot * num_games + game, reviews[inside])
        # slot j (j >= 1) holds week first_week + j - 1, so the cumulative row j counts every
        # week < first_week + j
        pos = np.cumsum(pos.reshape(weeks + 1, num_games), axis=0)
        tot = np.cumsum(tot.reshape(weeks + 1, num_games), axis=0)
        return cls(first_week, pos, tot)

    def row(self, week: np.ndarray) -> np.ndarray:
        """Row of each week's lookup (clipped to the table)."""
        return np.clip(np.asarray(week) - self.first_week, 0, len(self.positives) - 1)


class PopularityCounter:
    """Builds a `PopularityTable` from reviews streamed in batches (one pass over
    `interactions`): weeks before `first_week` are summed into row 0, weeks after `last_week`
    are dropped."""

    def __init__(self, num_games: int, first_week: int, last_week: int) -> None:
        self.num_games = num_games
        self.first_week = first_week
        self.weeks = last_week - first_week + 1
        size = (self.weeks + 1) * num_games
        self._positives = np.zeros(size, dtype=np.int64)
        self._reviews = np.zeros(size, dtype=np.int64)

    def add(self, game: np.ndarray, seconds: np.ndarray, is_positive: np.ndarray) -> None:
        week = week_of(seconds)
        keep = (game >= 0) & (game < self.num_games) & (week < self.first_week + self.weeks)
        slot = np.clip(week[keep] - self.first_week + 1, 0, self.weeks)
        index = slot * self.num_games + game[keep]
        size = len(self._reviews)
        self._reviews += np.bincount(index, minlength=size)
        self._positives += np.bincount(index[is_positive[keep]], minlength=size)

    def table(self) -> PopularityTable:
        shape = (self.weeks + 1, self.num_games)
        return PopularityTable(
            self.first_week,
            np.cumsum(self._positives.reshape(shape), axis=0),
            np.cumsum(self._reviews.reshape(shape), axis=0),
        )


def compute_features(
    history: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    seconds: np.ndarray,
    games: GameTable,
    popularity: PopularityTable,
) -> np.ndarray:
    """float32 [groups, candidates, len(FEATURES)] (NaN = missing). `history` int64 [b, h]
    (most recent first, padding < FIRST_ID), `candidates` int64 [b, k] and `scores` [b, k]
    (best first; -inf = no candidate), `seconds` int64 [b]: time t of each group."""
    b, k = candidates.shape
    out = np.full((b, k, len(FEATURES)), np.nan, dtype=np.float32)
    col = {name: i for i, name in enumerate(FEATURES)}
    valid = (history >= FIRST_ID) & (history < games.num_games)
    hist = np.where(valid, history, 0)
    n_hist = valid.sum(axis=1)
    has_hist = n_hist > 0
    denom = np.maximum(n_hist, 1)[:, None]
    cand = np.clip(candidates, 0, games.num_games - 1)

    # retrieval: positions only (scale-free)
    finite = np.isfinite(scores)
    n_cand = finite.sum(axis=1)
    rank = np.broadcast_to(np.arange(k, dtype=np.float32), (b, k))
    out[..., col["rank"]] = rank
    out[..., col["score_percentile"]] = 1 - rank / np.maximum(n_cand - 1, 1)[:, None]
    top = np.where(finite[:, :1], scores[:, :1], 0)
    lowest = np.min(np.where(finite, scores, np.inf), axis=1, keepdims=True)
    spread = np.where(np.isfinite(lowest), top - lowest, 0)
    out[..., col["score_gap"]] = (scores - top) / np.maximum(spread, 1e-9)

    # candidate, as of t (the weeks before t's week)
    row = popularity.row(week_of(seconds))[:, None]
    lifetime = popularity.positives[row, cand].astype(np.float32)
    reviews = popularity.reviews[row, cand].astype(np.float32)
    out[..., col["log_positives"]] = np.log1p(lifetime)
    for name, weeks in RECENT_WEEKS.items():
        earlier = popularity.positives[np.maximum(row - weeks, 0), cand]
        out[..., col[name]] = np.log1p(lifetime - earlier)
    out[..., col["positive_ratio"]] = (lifetime + 1) / (reviews + 2)
    out[..., col["is_free"]] = games.is_free[cand]
    out[..., col["log_price"]] = games.log_price[cand]
    day = (seconds / SECONDS_PER_DAY).astype(np.float32)[:, None]
    out[..., col["age_days"]] = day - games.release_day[cand]

    # affinity: H's profiles vs the candidate
    cand_tags = games.tags[cand]  # [b, k, tags]
    hist_tags = games.tags[hist] * valid[..., None]  # [b, h, tags]
    profile = hist_tags.sum(axis=1)
    profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
    out[..., col["tag_cosine"]] = np.einsum("bkt,bt->bk", cand_tags, profile)
    for name, ids in (("developer_share", games.developer), ("publisher_share", games.publisher)):
        same = (ids[cand][:, :, None] == ids[hist][:, None, :]) & valid[:, None, :]
        same &= ids[cand][:, :, None] >= 0
        out[..., col[name]] = same.sum(axis=2) / denom
    for name, table in (("genre_overlap", games.genres), ("category_overlap", games.categories)):
        hist_share = (table[hist] * valid[..., None]).sum(axis=1) / denom  # [b, labels]
        cand_labels = table[cand]  # [b, k, labels]
        overlap = np.einsum("bkl,bl->bk", cand_labels, hist_share)
        out[..., col[name]] = overlap / np.maximum(cand_labels.sum(axis=2), 1)

    # no history: user-side features are undefined; no candidate: everything is
    user_side = [col[f] for f in FAMILIES["affinity"]]
    if not has_hist.all():
        idx = np.flatnonzero(~has_hist)
        out[np.ix_(idx, np.arange(k), np.array(user_side))] = np.nan
    out[~finite] = np.nan
    return out
