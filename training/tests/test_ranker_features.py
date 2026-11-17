from datetime import datetime

import numpy as np
import pytest

from steam_training.data import Ragged
from steam_training.ranker.data import (
    cap_per_user,
    epoch_seconds,
    from_epoch_seconds,
    parse_release_day,
    positives_after,
)
from steam_training.ranker.features import (
    FEATURES,
    GameTable,
    PopularityCounter,
    compute_features,
    week_of,
)

WEEK = 7 * 86_400
NUM_GAMES = 8


def table(**overrides) -> GameTable:
    """Games 2..7: tags = one-hot of (game % 3); developer = game % 2; game 7 has no tags."""
    tags = np.zeros((NUM_GAMES, 3), dtype=np.float32)
    for g in range(2, 7):
        tags[g, g % 3] = 1
    values = dict(
        tags=tags,
        developer=np.array([-1, -1, 0, 1, 0, 1, 0, -1]),
        publisher=np.full(NUM_GAMES, -1),
        genres=np.eye(NUM_GAMES, 2, dtype=np.float32),
        categories=np.zeros((NUM_GAMES, 1), dtype=np.float32),
        is_free=np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32),
        log_price=np.log1p(np.arange(NUM_GAMES, dtype=np.float32)),
        release_day=np.arange(NUM_GAMES, dtype=np.float32) * 100,
    )
    values.update(overrides)
    return GameTable(**values)


def popularity(rows: list[tuple[int, int, bool]], first_week: int, last_week: int):
    counter = PopularityCounter(NUM_GAMES, first_week, last_week)
    game, week, positive = (np.array(x) for x in zip(*rows, strict=True))
    counter.add(game, week * WEEK + 3600, positive)
    return counter.table()


def features(history, candidates, scores, seconds, games=None, pop=None):
    games = games or table()
    pop = pop or popularity([(2, 0, True)], 0, 10)
    out = compute_features(
        np.array(history), np.array(candidates), np.array(scores, dtype=np.float32),
        np.array(seconds), games, pop,
    )  # fmt: skip
    return {name: out[..., i] for i, name in enumerate(FEATURES)}


def test_popularity_counts_only_weeks_before_t():
    pop = popularity(
        [(2, 1, True), (2, 1, False), (2, 5, True), (2, 9, True), (3, 0, True)],
        first_week=2,
        last_week=10,
    )
    # weeks before first_week land in row 0
    assert pop.positives[0, 2] == 1 and pop.reviews[0, 2] == 2 and pop.positives[0, 3] == 1
    assert pop.positives[pop.row(5), 2] == 1  # week 5 itself not counted yet
    assert pop.positives[pop.row(6), 2] == 2
    assert pop.positives[pop.row(100), 2] == 3  # beyond the table: the last row
    f = features([[3]], [[2]], [[1.0]], [9 * WEEK + 10], pop=pop)
    assert f["log_positives"][0, 0] == pytest.approx(np.log1p(2))
    assert f["log_positives_30d"][0, 0] == pytest.approx(np.log1p(1))  # weeks 5..8
    assert f["positive_ratio"][0, 0] == pytest.approx((2 + 1) / (3 + 2))


def test_affinity():
    # history: games 2 (tag 2, dev 0) and 5 (tag 2, dev 1) and 3 (tag 0, dev 1); pad
    f = features([[2, 5, 3, 0]], [[4, 6, 7]], [[3.0, 2.0, 1.0]], [WEEK * 5])
    # candidate 4 (tag 1, dev 0), 6 (tag 0, dev 0), 7 (no tags, no dev)
    profile = np.array([1, 0, 2]) / np.sqrt(5)
    assert f["tag_cosine"][0].tolist() == pytest.approx([0, profile[0], 0])
    assert f["developer_share"][0].tolist() == pytest.approx([1 / 3, 1 / 3, 0])
    # genres = eye(8, 2): only games 0 and 1 have one, so no overlap here
    assert f["genre_overlap"][0].tolist() == pytest.approx([0, 0, 0])
    assert f["age_days"][0, 0] == pytest.approx(35 - 400)


def test_retrieval_features_are_scale_free():
    a = features([[2]], [[3, 4, 5]], [[0.9, 0.5, 0.1]], [WEEK * 5])
    b = features([[2]], [[3, 4, 5]], [[90.0, 50.0, 10.0]], [WEEK * 5])
    for name in ("rank", "score_percentile", "score_gap"):
        assert a[name][0].tolist() == pytest.approx(b[name][0].tolist())
    assert a["score_gap"][0].tolist() == pytest.approx([0, -0.5, -1])
    assert a["score_percentile"][0].tolist() == pytest.approx([1, 0.5, 0])


def test_no_history_and_missing_candidates():
    f = features([[0, 1]], [[3, 4]], [[0.5, -np.inf]], [WEEK * 5])
    assert np.isnan(f["tag_cosine"][0, 0]) and np.isnan(f["developer_share"][0, 0])
    assert not np.isnan(f["log_positives"][0, 0])  # candidate features stay
    assert all(np.isnan(f[name][0, 1]) for name in FEATURES)  # no candidate in slot 1


def test_cap_per_user():
    users = np.array([1] * 30 + [2] * 3)
    kept = cap_per_user(users, cap=5, seed=0)
    assert (users[kept] == 1).sum() == 5 and (users[kept] == 2).sum() == 3
    assert (np.diff(kept) > 0).all()


def test_positives_after_drops_the_target():
    # user 0: games 5 (t=10), 6 (t=20), 7 (t=30); user 1: game 5 (t=15)
    later = positives_after(
        np.array([0, 0, 0, 1]),
        np.array([10, 20, 30, 15]),
        np.array([5, 6, 7, 5]),
        row_user=np.array([0, 0, 1]),
        row_seconds=np.array([10, 20, 15]),
        row_target=np.array([5, 6, 5]),
    )
    assert [later.row(i) for i in range(3)] == [[6, 7], [7], []]
    assert isinstance(later, Ragged)


def test_dates():
    assert parse_release_day("Oct 9, 2020") == pytest.approx(18544)
    assert parse_release_day("9 Oct, 2020") == pytest.approx(18544)
    assert np.isnan(parse_release_day("Coming soon"))
    cutoff = datetime(2026, 5, 1, 20, 13, 50)
    assert from_epoch_seconds(epoch_seconds(cutoff)) == cutoff
    assert week_of(epoch_seconds(datetime(1970, 1, 8))) == 1


def test_online_export_game_side_matches_the_features():
    """`ranker.export.game_arrays` (serving's game side) = compute_features' candidate columns
    at the export time; tags / genres as CSR rows of the published catalog order."""
    from steam_training.ranker.export import game_arrays

    games = table()
    pop = popularity([(2, 1, True), (2, 3, False), (4, 2, True), (5, 8, True)], 0, 10)
    rows = np.array([5, 2, 4, 3])  # the published catalog's game_idx, in its row order
    seconds = 9 * WEEK + 10
    exported = game_arrays(games, pop, rows, seconds)
    f = features([[3]], [rows.tolist()], [[4.0, 3.0, 2.0, 1.0]], [seconds], pop=pop)
    for i, name in enumerate(["log_positives", "log_positives_30d", "log_positives_90d"]):
        np.testing.assert_array_equal(exported["game_popularity"][:, i], f[name][0])
    np.testing.assert_array_equal(exported["game_popularity"][:, 3], f["positive_ratio"][0])
    np.testing.assert_array_equal(exported["game_is_free"], games.is_free[rows])
    offsets, values = exported["game_tag_offsets"], exported["game_tag_values"]
    for r, game in enumerate(rows):
        assert (
            values[offsets[r] : offsets[r + 1]].tolist()
            == np.flatnonzero(games.tags[game]).tolist()
        )
    assert exported["game_idx"].tolist() == rows.tolist()
