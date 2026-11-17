from datetime import datetime

import numpy as np
import pytest

from steam_training.data import Ragged, split_window
from steam_training.ranker.lists import (
    ListAccumulator,
    UserReviews,
    snapshot_lists,
    snapshots,
)

DAY = 86_400


def reviews(rows):
    """(user, day, game, positive) rows."""
    u, d, g, p = (np.array(x) for x in zip(*rows, strict=True))
    return UserReviews.build(u, d * DAY, g, p.astype(bool))


def test_snapshot_lists():
    r = reviews(
        [
            (1, 1, 2, True),
            (1, 2, 3, False),
            (1, 3, 4, True),
            (1, 10, 5, True),  # relevant at t = day 8 (horizon 7 days)
            (1, 11, 3, True),  # reviewed before t: not relevant
            (1, 20, 6, True),  # after the horizon
            (2, 9, 7, True),  # no liked game before t: no list
            (3, 1, 2, True),  # nothing in the horizon: no list
        ]
    )
    lists = snapshot_lists(r, t=8 * DAY, horizon=7 * DAY, length=3)
    assert lists.user_id.tolist() == [1]
    assert lists.history.tolist() == [[4, 2, 0]]  # most recent first
    assert lists.reviewed.row(0) == [2, 3, 4]
    assert lists.relevant.row(0) == [5]


def test_list_metrics():
    acc = ListAccumulator([2, 3])
    # list 0: relevant {5, 9}, ranked [5, 7, 9]; list 1: relevant {4}, ranked [7, 8, -1]
    acc.add(np.array([[5, 7, 9], [7, 8, -1]]), Ragged.from_lists([[5, 9], [4]]))
    m = acc.metrics()
    assert m.lists == 2
    assert m.precision[2] == pytest.approx((1 / 2 + 0) / 2)
    assert m.recall[2] == pytest.approx((1 / 2 + 0) / 2)
    assert m.recall[3] == pytest.approx((2 / 2 + 0) / 2)
    # AP@3 of list 0: (1/1 + 2/3) / min(3, 2)
    assert m.map[3] == pytest.approx(((1 + 2 / 3) / 2 + 0) / 2)


def test_snapshots_keep_the_latest():
    assert snapshots(0, 30, 7, 10) == [0, 7, 14, 21]
    assert snapshots(0, 30, 7, 2) == [14, 21]


def test_holdout_split_is_counted_back_from_the_newest_row():
    ts = np.array(["2026-01-01", "2026-03-01", "2026-03-29"], dtype="datetime64[us]")
    cutoff, end, name = split_window(ts, 0.1, 0.0, "train", holdout_weeks=6, test_weeks=2)
    assert (cutoff, end, name) == (datetime(2026, 2, 15), datetime(2026, 3, 15), "validation")
    cutoff, end, name = split_window(ts, 0.1, 0.0, "train+validation", 6, 2)
    assert (cutoff, end, name) == (datetime(2026, 3, 15), None, "test")
