from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from steam_eda.ranking import (
    FEATURES,
    GameArrays,
    Groups,
    ProxyRetriever,
    Sequences,
    WeeklyPopularity,
    build_groups,
    group_auc,
    group_features,
    group_sizes,
    hits,
    ranking_metrics,
    weekly_counts,
)

T1 = datetime(2026, 5, 1)
T2 = datetime(2026, 8, 1)


def games_df(n: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [100 + i for i in range(n)],
            "tag_names": [["Horror", "Zombies"] if i % 2 else ["Racing", "Cars"] for i in range(n)],
            "tag_weights": [[200, 100]] * n,
            "game_genres": [["Action"] if i % 2 else ["Racing"] for i in range(n)],
            "game_categories": [["Single-player"]] * n,
            "game_developers": [[f"dev{i % 3}"] for i in range(n)],
            "game_publishers": [[f"pub{i % 2}"] for i in range(n)],
            "game_is_free": [i == 0 for i in range(n)],
            "game_price": [0.0 if i == 0 else 9.99 for i in range(n)],
            "release_date": [date(2020, 1, 1) + timedelta(days=100 * i) for i in range(n)],
        }
    )


def reviews_df(rows: list[tuple[int, int, bool, datetime]]) -> pl.DataFrame:
    u, g, p, t = zip(*rows, strict=True)
    return pl.DataFrame(
        {
            "user_id": list(u),
            "game_id": list(g),
            "is_positive": list(p),
            "reviewed_at": list(t),
            "language": ["english"] * len(rows),
            "playtime_at_review": [600] * len(rows),
        }
    )


def day(m: int, d: int) -> datetime:
    return datetime(2026, m, d, 12)


@pytest.fixture
def games() -> GameArrays:
    return GameArrays.build(games_df())


def test_game_arrays(games):
    assert games.n == 6
    np.testing.assert_allclose(np.linalg.norm(games.tags, axis=1), 1, rtol=1e-6)
    assert games.is_free.tolist() == [True] + [False] * 5
    assert games.developer[0] == games.developer[3] != games.developer[1]


def test_groups_warm_only_cap_and_split(games):
    rows = [
        # user 1: positive before t1, then three positives after (one after T2)
        (1, 100, True, day(4, 1)),
        (1, 101, True, day(5, 10)),
        (1, 102, False, day(6, 1)),
        (1, 103, True, day(6, 10)),
        (1, 104, True, day(9, 1)),
        # user 2: first positive after t1 is not warm; the second one is
        (2, 100, False, day(4, 1)),
        (2, 101, True, day(5, 10)),
        (2, 102, True, day(5, 20)),
    ]
    seq = Sequences.build(reviews_df(rows), games.gid2gi)
    train, test = build_groups(seq, T1, T2, cap=20)
    targets = sorted((int(seq.user[i]), int(seq.gi[i])) for _, _, i in np.r_[train.rows, test.rows])
    assert targets == [(1, 1), (1, 3), (1, 4), (2, 2)]
    assert [int(seq.gi[i]) for _, _, i in test.rows] == [4]
    ts = seq.ts[train.rows[:, 2]]
    assert (np.diff(ts) >= 0).all()  # train sorted by time
    train, test = build_groups(seq, T1, T2, cap=1)
    assert len(train) + len(test) == 2  # one per user


def test_weekly_popularity_excludes_current_week(games):
    rows = [(u, 100, True, day(5, 4)) for u in range(3)] + [(9, 100, False, day(5, 20))]
    pop = WeeklyPopularity.build(weekly_counts(reviews_df(rows).lazy()), games.gid2gi)
    wk = int(day(5, 4).timestamp()) // (7 * 86_400)
    gi = np.array([0, 1])
    assert pop.positives(gi, wk).tolist() == [0, 0]  # same week: not yet counted
    assert pop.positives(gi, wk + 1).tolist() == [3, 0]
    assert pop.reviews(gi, wk + 10).tolist() == [4, 0]
    assert pop.window(wk + 1, 1, games.n)[0] == 3


def build(rows, games):
    seq = Sequences.build(reviews_df(rows), games.gid2gi)
    pop = WeeklyPopularity.build(weekly_counts(reviews_df(rows).lazy()), games.gid2gi)
    emb = np.eye(games.n, 4, dtype=np.float32)
    return seq, pop, emb


BASE = [
    (1, 100, True, day(3, 1)),
    (1, 101, True, day(3, 2)),
    (1, 102, False, day(3, 3)),
    (1, 103, True, day(6, 1)),  # target
]


def target_group(seq: Sequences) -> Groups:
    i = int(np.flatnonzero((seq.user == 1) & (seq.gi == 3))[0])
    a, b = seq.bounds[0], seq.bounds[1]
    return Groups(np.array([[a, b, i]]))


def features(rows, games, cands=(3, 4, 5), keep_missing=False):
    seq, pop, emb = build(rows, games)
    groups = target_group(seq)
    c = np.array([list(cands)])
    s = np.array([[0.9, 0.5, 0.1][: len(cands)]], dtype=np.float32)
    return group_features(seq, groups, c, s, games, pop, emb, {}, history=1,
                          keep_missing=keep_missing)  # fmt: skip


def test_features_ignore_reviews_from_t_on(games):
    x, y, _ = features(BASE, games)
    later = BASE + [
        (1, 104, False, day(7, 1)),
        (2, 105, True, day(6, 1)),
        (3, 103, True, day(9, 1)),
    ]
    x2, y2, _ = features(later, games)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    assert x.shape == (3, len(FEATURES)) and y.tolist() == [1, 0, 0]


def test_features_values(games):
    x, _, _ = features(BASE, games)
    f = dict(zip(FEATURES, x[0], strict=True))  # the target, game 3 (Horror, dev0)
    assert f["n_pos"] == 2 and f["n_neg"] == 1
    assert f["hist_gt_last"] == 1  # 2 positives > history=1
    assert f["dev_share"] == pytest.approx(0.5)  # games 0 (dev0) and 1 (dev1)
    assert f["neg_tag_cos"] == pytest.approx(0.0)  # game 2 is Racing, game 3 Horror
    assert f["tag_cos_last"] == pytest.approx(1.0)  # last positive = game 1, Horror
    assert f["days_since_last"] == pytest.approx((day(6, 1) - day(3, 3)).days)
    assert f["log_pos"] == pytest.approx(0.0)  # nobody reviewed game 3 before


def test_later_positives_dropped_and_missing_groups(games):
    rows = BASE + [(1, 104, True, day(7, 1))]
    x, y, _ = features(rows, games)
    assert len(y) == 2  # game 4 (a later positive) is not a negative
    _, y, _ = features(rows, games, cands=(4, 5))
    assert len(y) == 0  # no positive among the candidates: dropped for training
    _, y, _ = features(rows, games, cands=(4, 5), keep_missing=True)
    assert y.tolist() == [0]


def test_retriever_excludes_seen_games(games):
    seq, pop, emb = build(BASE, games)
    retriever = ProxyRetriever.build(
        emb, np.full(games.n, 10), pop, [datetime(2026, m, 1) for m in (5, 6, 7)], 20, 1
    )
    cands, scores = retriever.retrieve(seq, target_group(seq), k=3)
    assert cands.shape == (1, 3)
    assert not {0, 1, 2} & set(cands[0].tolist())
    assert (np.diff(scores[0]) <= 0).all()
    assert hits(seq, target_group(seq), cands).sum() == 1


def test_metrics_and_group_auc():
    qid = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    label = np.array([0, 1, 0, 1, 0, 0, 0, 0])
    score = np.array([3, 2, 1, 1, 2, 3, 1, 2], dtype=float)
    assert group_sizes(qid).tolist() == [3, 3, 2]
    m = ranking_metrics(score, label, qid, ks=(1, 2))
    assert m["recall@1"] == pytest.approx(0)
    assert m["recall@2"] == pytest.approx(1 / 3)  # group 1 has its positive at rank 3
    assert m["ndcg@2"] == pytest.approx(1 / np.log2(3) / 3)
    auc, n = group_auc(score, label, qid)
    assert n == 2 and auc == pytest.approx((0.5 + 0.0) / 2)
