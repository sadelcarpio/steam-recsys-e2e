"""Per-user list evaluation (spec 9): the batch path's view of a ranker.

Batch inference writes one list per user and run; the user then reviews games until the next
run. So, at each snapshot T (weekly, inside the test window), every sampled user with a liked
game before T gets one list built from their history before T (the retriever's top candidates,
then retrieval order or ranker order), and the relevant games are the ones they review
positively in [T, T + horizon) that they had not reviewed before T. Several games can be
relevant, so precision@k, recall@k (hits / min(k, relevant)) and MAP@k are meaningful here,
unlike the per-review groups (one relevant game each).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from steam_training.contracts import FIRST_ID, ListMetrics
from steam_training.data import Ragged


@dataclass(frozen=True)
class UserReviews:
    """Every review of a sample of users, sorted by (user, time)."""

    user_id: np.ndarray  # int64
    seconds: np.ndarray  # int64
    game: np.ndarray  # int64 game_idx
    positive: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.game)

    @classmethod
    def build(cls, user_id, seconds, game, positive) -> UserReviews:
        order = np.lexsort((seconds, user_id))
        return cls(user_id[order], seconds[order], game[order], positive[order])

    @property
    def bounds(self) -> np.ndarray:
        """Start of each user's slice, plus len(self) at the end."""
        u = self.user_id
        return np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1, len(u)].astype(np.int64)


@dataclass(frozen=True)
class SnapshotLists:
    """The lists of one snapshot: one row per user with a liked game before T and at least one
    relevant game in [T, T + horizon)."""

    user_id: np.ndarray
    history: np.ndarray  # int64 [users, length], most recent first, 0 = padding
    reviewed: Ragged  # every game reviewed before T (excluded from the candidates)
    relevant: Ragged

    def __len__(self) -> int:
        return len(self.user_id)


def snapshot_lists(reviews: UserReviews, t: int, horizon: int, length: int) -> SnapshotLists:
    """Lists at time `t` (seconds), relevant games until `t + horizon` (seconds)."""
    users, histories, reviewed, relevant = [], [], [], []
    bounds = reviews.bounds
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        sec, game, pos = reviews.seconds[a:b], reviews.game[a:b], reviews.positive[a:b]
        before = sec < t
        liked = game[before & pos][::-1][:length]
        if not len(liked):
            continue
        seen = game[before]
        later = game[(sec >= t) & (sec < t + horizon) & pos]
        later = np.unique(later[~np.isin(later, seen)])
        if not len(later):
            continue
        row = np.zeros(length, dtype=np.int64)
        row[: len(liked)] = liked
        users.append(reviews.user_id[a])
        histories.append(row)
        reviewed.append(np.unique(seen).tolist())
        relevant.append(later.tolist())
    return SnapshotLists(
        user_id=np.array(users, dtype=np.int64),
        history=np.array(histories, dtype=np.int64).reshape(-1, length),
        reviewed=Ragged.from_lists(reviewed),
        relevant=Ragged.from_lists(relevant),
    )


@dataclass
class ListAccumulator:
    """Sums of per-list precision / recall / AP at each k, over (user, snapshot) lists."""

    ks: list[int]
    lists: int = 0

    def __post_init__(self) -> None:
        self.precision = dict.fromkeys(self.ks, 0.0)
        self.recall = dict.fromkeys(self.ks, 0.0)
        self.ap = dict.fromkeys(self.ks, 0.0)

    def add(self, ordered: np.ndarray, relevant: Ragged) -> None:
        """`ordered` int64 [lists, n]: each list's games, best first (-1 = no game)."""
        n = len(ordered)
        if not n:
            return
        owner = np.repeat(np.arange(n), np.diff(relevant.offsets))
        width = int(ordered.max(initial=0)) + 2
        hit = np.isin(
            np.arange(n)[:, None] * width + (ordered + 1),
            owner * width + (relevant.values + 1),
        ) & (ordered >= FIRST_ID)
        size = np.diff(relevant.offsets)
        cumulative = np.cumsum(hit, axis=1)
        position = np.arange(1, ordered.shape[1] + 1)
        for k in self.ks:
            kk = min(k, ordered.shape[1])
            hits = cumulative[:, kk - 1]
            denom = np.minimum(k, size)
            self.precision[k] += float((hits / k).sum())
            self.recall[k] += float((hits / denom).sum())
            precision_at = cumulative[:, :kk] / position[:kk]
            self.ap[k] += float(((precision_at * hit[:, :kk]).sum(axis=1) / denom).sum())
        self.lists += n

    def metrics(self) -> ListMetrics:
        n = max(self.lists, 1)
        return ListMetrics(
            lists=self.lists,
            precision={k: v / n for k, v in self.precision.items()},
            recall={k: v / n for k, v in self.recall.items()},
            map={k: v / n for k, v in self.ap.items()},
        )


def snapshots(start: int, end: int, horizon: int, max_snapshots: int) -> list[int]:
    """Snapshot times start, start + horizon, ... whose horizon ends by `end` (at most
    `max_snapshots`, the latest kept when there are more)."""
    times = list(range(start, end - horizon + 1, horizon))
    return times[-max_snapshots:] if max_snapshots else []


def evaluate_lists(
    reviews: UserReviews,
    times: list[int],
    horizon: int,
    length: int,
    ks: list[int],
    rank: Callable[[SnapshotLists, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[ListMetrics, ListMetrics]:
    """(retrieval order, ranker order) metrics. `rank(lists, t)` returns the candidates in
    retrieval order and in ranker order (int64 [lists, n], -1 = no game)."""
    retrieval, ranker = ListAccumulator(ks), ListAccumulator(ks)
    for t in times:
        lists = snapshot_lists(reviews, t, horizon, length)
        if not len(lists):
            continue
        by_retrieval, by_ranker = rank(lists, t)
        retrieval.add(by_retrieval, lists.relevant)
        ranker.add(by_ranker, lists.relevant)
    return retrieval.metrics(), ranker.metrics()
