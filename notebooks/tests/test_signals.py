from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from steam_eda import (
    auc,
    binned_mean,
    consecutive_pairs,
    gini,
    jaccard,
    lag_similarity,
    profile_fit,
    shuffle_games,
    shuffle_order,
    smoothed_ratio,
    spearman,
    user_concentration,
)
from steam_eda.nlp import knn, tfidf_embed

T0 = datetime(2025, 1, 1)


def reviews(rows: list[tuple[int, int]]) -> pl.DataFrame:
    """(user, game) rows, in time order per user."""
    return pl.DataFrame(
        {
            "review_id": list(range(1, len(rows) + 1)),
            "user_id": [u for u, _ in rows],
            "game_id": [g for _, g in rows],
            "reviewed_at": [T0 + timedelta(days=i) for i in range(len(rows))],
        }
    )


LABELS = pl.DataFrame(
    {
        "game_id": [1, 2, 3, 4],
        "tags": [["horror", "zombies"], ["horror"], ["racing"], []],
    }
)


def test_jaccard() -> None:
    df = pl.DataFrame({"a": [["x", "y"], [], ["x"]], "b": [["y", "z"], [], ["x"]]})
    assert df.select(jaccard(pl.col("a"), pl.col("b"))).to_series().to_list() == [1 / 3, None, 1.0]


def test_shuffle_games_keeps_marginals() -> None:
    df = reviews([(u, g) for u in range(20) for g in range(u % 5 + 1)])
    s = shuffle_games(df, seed=3)
    assert (
        s.group_by("user_id")
        .len()
        .sort("user_id")
        .equals(df.group_by("user_id").len().sort("user_id"))
    )
    assert sorted(s["game_id"]) == sorted(df["game_id"])
    assert not s["game_id"].equals(df["game_id"])


def test_user_concentration() -> None:
    df = reviews([(1, 1), (1, 2), (1, 3), (2, 1), (3, 4)])
    out = user_concentration(df, LABELS, "tags").sort("user_id")
    one = out.row(0, named=True)
    # games 1, 2 carry horror (2 of 3 labelled games); counts: horror 2, zombies 1, racing 1
    assert one["n_games"] == 3 and one["top_label"] == "horror"
    assert one["top_share"] == pytest.approx(2 / 3)
    assert one["entropy"] == pytest.approx(1.5)
    assert one["effective_labels"] == pytest.approx(2**1.5)
    assert out["user_id"].to_list() == [1, 2]  # user 3 only has an unlabelled game
    assert user_concentration(df, LABELS, "tags", min_reviews=2)["user_id"].to_list() == [1]


def test_profile_fit_leave_one_out() -> None:
    df = reviews([(1, 1), (1, 2), (1, 3), (2, 1)])
    fit = profile_fit(df, df, LABELS, "tags").sort("review_id")["fit"].to_list()
    # review 1 (horror, zombies) vs others {2: horror, 3: racing}: horror 1/2, zombies 0/2
    assert fit[0] == pytest.approx(0.25)
    assert fit[1] == pytest.approx(0.5)  # horror: game 1 of the 2 others
    assert fit[2] == pytest.approx(0.0)
    assert fit[3] is None  # no other history game


def test_profile_fit_of_outside_rows() -> None:
    hist = reviews([(1, 1), (1, 2)])
    scored = pl.DataFrame({"review_id": [-1, -2], "user_id": [1, 1], "game_id": [2, 3]})
    fit = profile_fit(hist, scored, LABELS, "tags").sort("review_id")["fit"].to_list()
    assert fit == [pytest.approx(0.0), pytest.approx(1.0)]  # racing: none; horror: both


def test_lag_similarity_detects_order() -> None:
    # each user alternates two blocks: horror for 25 reviews then racing, so near lags match
    rows = [(u, 1 if (i // 25) % 2 == 0 else 3) for u in range(5) for i in range(60)]
    df = reviews(rows)
    lags = lag_similarity(df, LABELS, "tags", max_lag=10)
    assert lags["lag"].to_list() == list(range(1, 11))
    assert lags["pairs"].n_unique() == 1  # same targets at every lag
    sims = lags["similarity"].to_list()
    assert sims[0] > sims[-1]
    shuffled = lag_similarity(shuffle_order(df, seed=1), LABELS, "tags", max_lag=10)
    assert shuffled["similarity"][0] < sims[0]


def test_shuffle_order_keeps_users_games_and_times() -> None:
    df = reviews([(u, g) for u in range(3) for g in [1, 2, 3, 4, 1, 2]])
    s = shuffle_order(df, seed=0)

    def games_per_user(d: pl.DataFrame) -> pl.DataFrame:
        return d.group_by("user_id").agg(pl.col("game_id").sort()).sort("user_id")

    assert games_per_user(s).equals(games_per_user(df))
    assert sorted(s["reviewed_at"]) == sorted(df["reviewed_at"])
    assert not s["game_id"].equals(df.sort("user_id", "reviewed_at")["game_id"])


def test_consecutive_pairs() -> None:
    out = consecutive_pairs(reviews([(1, 1), (1, 2), (2, 3)]), LABELS, "tags")
    assert out.height == 1
    assert out["gap_days"][0] == pytest.approx(1.0)
    assert out["similarity"][0] == pytest.approx(0.5)


def test_binned_mean_keeps_ties_together() -> None:
    df = pl.DataFrame({"x": [0] * 70 + list(range(1, 31)), "y": [0.0] * 70 + [1.0] * 30})
    out = binned_mean(df, "x", "y", bins=10)
    assert out["rows"][0] == 70 and out["mean"][0] == 0.0
    assert out["rows"].sum() == 100
    assert out["mean"][-1] == 1.0
    assert binned_mean(pl.DataFrame({"x": [None], "y": [1.0]}), "x", "y").height == 0


def test_auc_spearman_gini_ratio() -> None:
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([1, 1], [0.1, 0.2]) is None
    assert auc(np.array([0, 1, np.nan]), np.array([0.1, 0.9, 0.5])) == 1.0
    assert spearman(pl.Series([1.0, 2, 3, None]), pl.Series([3.0, 2, 1, 5])) == pytest.approx(-1.0)
    assert gini(pl.Series([1, 1, 1, 1])) == pytest.approx(0.0)
    assert gini(pl.Series([0, 0, 0, 10])) == pytest.approx(0.75)
    ratio = pl.select(smoothed_ratio(pl.lit(0), pl.lit(0), prior=0.8)).item()
    assert ratio == pytest.approx(0.8)


def test_knn_and_tfidf() -> None:
    emb = tfidf_embed(["zombie horror night", "zombie horror fear", "car racing track",
                       "car racing drift", "puzzle blocks calm"])  # fmt: skip
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    nn = knn(emb, k=1)
    assert nn[:4, 0].tolist() == [1, 0, 3, 2]
    assert (nn[:, 0] != np.arange(5)).all()
