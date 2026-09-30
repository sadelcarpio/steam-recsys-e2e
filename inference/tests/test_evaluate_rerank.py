import json
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest
from conftest import BUCKET, FakeSource, ReverseLlm

from steam_inference.evaluate_rerank import (
    compare_orders,
    evaluate_rerank,
    format_table,
    hit_rates,
    load_cohort,
    sample_cohort,
    to_micros,
)

NOW = datetime(2024, 7, 1, tzinfo=UTC)
AFTER = datetime(2024, 6, 1)  # the champion's cutoff is 2024-05-01 (conftest.make_metadata)
CUTOFF = to_micros(datetime(2024, 5, 1))


@pytest.fixture
def later_marts(marts):
    """Reviews after the cutoff: 101 and 102 like a new game; 103 has too few reviews before;
    104's later review is negative. 101 also gets a user_features row after the cutoff."""
    later = [
        (101, 12, True),
        (102, 2, True),
        (103, 5, True),
        (104, 20, False),
    ]
    rows = [
        {"user_id": u, "game_idx": g, "is_positive": p, "timestamp": AFTER} for u, g, p in later
    ]
    interactions = marts["interactions"]
    marts["interactions"] = pa.concat_tables(
        [interactions, pa.Table.from_pylist(rows, schema=interactions.schema)]
    )
    features = marts["user_features"]
    extra = [{"user_id": 101, "timestamp": AFTER, "games_reviewed_positive": [12, 7, 6, 5, 3]}]
    marts["user_features"] = pa.concat_tables(
        [features, pa.Table.from_pylist(extra, schema=features.schema)]
    )
    return marts


def test_hit_rates():
    orders = [[2, 1, 0], [0, 1, 2]]
    assert hit_rates(orders, [{2}, {2}], cuts=(1, 3)) == {"hit@1": 0.5, "hit@3": 1.0}
    assert hit_rates([], [], cuts=(1,)) == {"hit@1": 0.0}


def test_compare_orders_includes_retrieval_llm_and_blends():
    rates = compare_orders([[4, 3, 2, 1, 0, 5]], [{5}], weights=(0.5,))
    assert set(rates) == {"retrieval", "llm", "blend_0.5"}
    # position 5 is 6th in every order: no hit at 5, a hit at 10
    assert all(r == {"hit@5": 0.0, "hit@10": 1.0} for r in rates.values())


def test_cohort_is_active_after_the_cutoff(later_marts):
    source = FakeSource(later_marts)
    cohort = sample_cohort(source, 7, CUTOFF, min_reviews=6, size=10, seed=0)
    assert cohort.tolist() == [101, 102]
    one = sample_cohort(source, 7, CUTOFF, min_reviews=6, size=1, seed=0)
    assert (
        len(one) == 1
        and one.tolist() == sample_cohort(source, 7, CUTOFF, min_reviews=6, size=1, seed=0).tolist()
    )


def test_cohort_history_is_as_of_the_cutoff(later_marts):
    source = FakeSource(later_marts)
    snapshots = {"interactions": 7, "user_features": 7}
    users, reviews, after = load_cohort(source, snapshots, np.array([101, 102]), CUTOFF)
    assert users.user_id.tolist() == [101, 102]
    assert users.history[0].tolist() == [7, 6, 5, 3, 2]  # not the row after the cutoff
    assert users.review_count.tolist() == [7, 6]  # the later reviews are not excluded
    assert 12 not in reviews.game_idx[: users.review_count[0]].tolist()
    assert after == {101: {12}, 102: {2}}


def test_evaluation_report_is_written(settings, later_marts, store, champion):
    report = evaluate_rerank(
        settings, FakeSource(later_marts), store, ReverseLlm(), users=10, now=NOW
    )
    assert report.model_id == "abc123" and report.cutoff == datetime(2024, 5, 1)
    assert (report.sampled_users, report.evaluated_users, report.rank_fallbacks) == (2, 2, 0)
    assert set(report.hit_rates) == {
        "retrieval",
        "llm",
        "blend_0.25",
        "blend_0.5",
        "blend_0.75",
    }
    assert all(0 <= v <= 1 for rates in report.hit_rates.values() for v in rates.values())
    keys = [o["Key"] for o in store.s3.list_objects_v2(Bucket=BUCKET)["Contents"]]
    key = "evaluation/rerank/abc123/20240701T000000Z.json"
    assert key in keys
    body = json.loads(store.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    assert body["evaluated_users"] == 2
    assert format_table(report).splitlines()[0].split() == ["order", "hit@5", "hit@10"]
