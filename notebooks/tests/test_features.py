from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from steam_eda import (
    binned_rate,
    categorical_rate,
    gini,
    lorenz,
    tag_profile_similarity,
    with_user_history,
)
from steam_eda.catalog import release_year

T0 = datetime(2026, 1, 1)


def test_user_history_uses_strictly_earlier_reviews() -> None:
    df = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "timestamp": [T0, T0 + timedelta(days=2), T0 + timedelta(days=2),
                          T0 + timedelta(days=3), T0],
            "is_positive": [True, False, True, True, False],
        }
    )  # fmt: skip
    out = with_user_history(df).sort("user_id", "timestamp", "is_positive")
    assert out["prior_reviews"].to_list() == [0, 1, 1, 3, 0]
    assert out["prior_positives"].to_list() == [0, 1, 1, 2, 0]
    assert out["prior_positive_rate"].to_list()[:4] == [None, 1.0, 1.0, pytest.approx(2 / 3)]
    assert out["days_since_prior_review"].to_list() == [None, 2.0, 2.0, 1.0, None]


def test_binned_rate_finds_a_monotone_signal() -> None:
    df = pl.DataFrame({"x": list(range(1000)), "is_positive": [i >= 500 for i in range(1000)]})
    rates = binned_rate(df, "x", bins=4)
    assert rates["rows"].to_list() == [250] * 4
    assert rates["rate"].to_list() == [0.0, 0.0, 1.0, 1.0]
    assert rates["lift"].to_list() == [0.0, 0.0, 2.0, 2.0]


def test_categorical_rate_explodes_lists_and_drops_rare() -> None:
    df = pl.DataFrame(
        {
            "tags": [[1, 2], [1], [2], [3]] * 50,
            "is_positive": [True, True, False, False] * 50,
        }
    )
    rates = categorical_rate(df, "tags", min_rows=60)
    assert rates["tags"].to_list() == [1, 2]  # tag 3 has 50 rows
    assert rates["rate"].to_list() == [1.0, 0.5]


def test_gini_and_lorenz() -> None:
    assert gini(pl.Series([5, 5, 5, 5])) == pytest.approx(0.0)
    assert gini(pl.Series([0, 0, 0, 100])) == pytest.approx(0.75)
    curve = lorenz(pl.Series([0, 0, 0, 100]))
    assert curve["share_of_total"][0] == 1.0 and curve["top_share"][-1] == 1.0


def test_tag_similarity() -> None:
    lists = pl.List(pl.Int64)
    sim = tag_profile_similarity(
        pl.Series([[1, 2], [], [1, 2, 3]], dtype=lists), pl.Series([[2, 3], [1], [1]], dtype=lists)
    )
    assert sim.to_list() == [pytest.approx(1 / 3), 0.0, pytest.approx(1 / 3)]
    empty = pl.Series([[]], dtype=lists)
    assert tag_profile_similarity(empty, empty).to_list() == [None]


def test_release_year() -> None:
    texts = pl.Series(["21 Aug, 2012", "Aug 21, 2012", "Q3 2026", "2026", "Coming soon", None])
    assert pl.select(release_year(pl.lit(texts))).to_series().to_list() == [
        2012, 2012, 2026, 2026, None, None
    ]  # fmt: skip


def test_single_feature_auc_and_report() -> None:
    from steam_eda import feature_report, single_feature_auc

    df = pl.DataFrame(
        {
            "perfect": [1.0, 2.0, 3.0, 4.0],
            "inverse": [4.0, 3.0, 2.0, 1.0],
            "flat": [1.0, 1.0, 1.0, 1.0],
            "sparse": [None, 2.0, None, 4.0],
            "label": [False, False, True, True],
        }
    )
    assert single_feature_auc(df, "perfect") == 1.0
    assert single_feature_auc(df, "inverse") == 0.0
    assert single_feature_auc(df, "flat") == 0.5
    assert single_feature_auc(df.filter(pl.col("label")), "perfect") is None
    report = feature_report(df, ["flat", "perfect", "inverse", "sparse"])
    assert report["feature"].to_list()[-1] == "flat"
    assert report.filter(pl.col("feature") == "sparse")["null_share"].item() == 0.5
