from datetime import datetime

import boto3
import numpy as np
import pytest
from moto import mock_aws

from steam_training.artifacts import ArtifactStore
from steam_training.config import RankerSettings
from steam_training.data import Ragged
from steam_training.pipeline import run_training
from steam_training.ranker.data import RankerRows
from steam_training.ranker.features import FEATURES
from steam_training.ranker.model import (
    Groups,
    build_groups,
    ranking_metrics,
    retrieval_score,
    target_positions,
)
from steam_training.ranker.pipeline import metrics_line, run_ranking
from tests.conftest import BUCKET, FakeSource, make_marts
from tests.test_ranker_features import popularity, table


def groups(labels: list[list[int]]) -> Groups:
    sizes = [len(g) for g in labels]
    offsets = np.r_[0, np.cumsum(sizes)]
    rank = np.concatenate([np.arange(n) for n in sizes]).astype(np.float32)
    x = np.zeros((offsets[-1], len(FEATURES)), dtype=np.float32)
    x[:, FEATURES.index("rank")] = rank
    return Groups(x, np.concatenate(labels).astype(np.int8), offsets, np.zeros(len(sizes), int))


def test_target_positions_and_metrics():
    g = groups([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    assert target_positions(retrieval_score(g), g).tolist() == [2, 0, -1]
    # a score that puts group 0's game first; ties keep retrieval order
    score = np.array([0, 0, 5, 1, 1, 1, 0, 0, 0], dtype=float)
    assert target_positions(score, g).tolist() == [0, 0, -1]
    m = ranking_metrics(np.array([2, 0, -1]), [1, 3])
    assert m.groups == 3
    assert m.recall == pytest.approx({1: 1 / 3, 3: 2 / 3})
    assert m.ndcg[3] == pytest.approx((1 / np.log2(4) + 1) / 3)


def test_build_groups_drops_later_positives_and_empty_slots():
    rows = RankerRows(
        user_id=np.array([1, 2]),
        seconds=np.array([7 * 86_400 * 5] * 2),
        history=np.array([[2, 0], [3, 2]]),
        target=np.array([4, 6]),
        reviewed=Ragged.from_lists([[2], [3, 2]]),
        later=Ragged.from_lists([[5], []]),  # user 1 likes game 5 later: not a negative
    )
    candidates = np.array([[5, 4, 6], [4, 5, 7]])
    scores = np.array([[3, 2, 1], [3, 2, -np.inf]], dtype=np.float32)
    g = build_groups(rows, candidates, scores, table(), popularity([(2, 0, True)], 0, 10), 2)
    assert g.sizes.tolist() == [2, 2]
    assert g.label.tolist() == [1, 0, 0, 0]
    assert g.has_target.tolist() == [True, False]
    assert g.history_size.tolist() == [1, 2]
    taken = g.take(np.array([1]))
    assert taken.sizes.tolist() == [2] and taken.label.tolist() == [0, 0]


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_rank_end_to_end(settings, store):
    source = FakeSource(make_marts(n_users=600, reviews_per_user=10, seed=1))
    run_training(settings.model_copy(update={"validation_fraction": 0.3}), source, store)
    ranker = RankerSettings(
        model_artifacts_bucket=BUCKET,
        ranker_id="rank1",
        retriever_id=settings.model_id,
        device="cpu",
        ranker_candidates=15,
        ranker_top_k=5,
        ranker_history_length=5,
        ranker_min_groups=20,
        ranker_min_data_in_leaf=5,
        ranker_max_rounds=50,
        ranker_early_stopping_rounds=10,
        eval_batch_size=128,
    )
    report = run_ranking(ranker, source, store)

    assert report.retriever_id == settings.model_id
    assert report.window.retriever_cutoff < report.window.t2 <= report.window.end
    assert report.window.test_groups > 0 and report.window.train_groups > 0
    assert 0 < report.candidate_recall <= 1
    assert set(report.test.ranker.recall) == {5, 10}
    assert set(report.ablation) == {"retrieval", "candidate", "affinity"}
    assert sum(report.importances.values()) == pytest.approx(1)
    assert "ranker_recall_at_k=" in metrics_line(report)
    # a miss for retrieval is a miss for the ranker: the ceiling bounds both
    assert report.test.ranker.recall[10] <= report.candidate_recall + 1e-9

    text, metadata = store.load_ranker("rank1")
    assert metadata.retriever_id == settings.model_id
    assert metadata.feature_names == FEATURES
    assert metadata.snapshots["game_details"] == 123
    assert "model_artifacts_bucket" not in metadata.hyperparameters
    assert "tree" in text.lower()
    assert store.read_ranker_report("rank1") == report

    import lightgbm as lgb

    booster = lgb.Booster(model_str=text)
    assert booster.num_feature() == len(FEATURES)


def test_ranker_settings_validation():
    base = {"model_artifacts_bucket": BUCKET, "ranker_id": "r1"}
    assert RankerSettings(**base).retriever_id == "champion"
    with pytest.raises(ValueError, match="RANKER_TOP_K"):
        RankerSettings(**base, ranker_candidates=10, ranker_top_k=20)
    with pytest.raises(ValueError, match="reserved"):
        RankerSettings(**{**base, "ranker_id": "champion"})


def test_popularity_mart_matches_the_interactions_count(marts):
    """The dbt mart and the fallback count the same weeks (Monday-based ISO weeks)."""
    from steam_training.ranker.data import _window_pass, popularity_from_mart
    from steam_training.ranker.features import PopularityCounter, week_of

    source = FakeSource(marts)
    ts = marts["interactions"]["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    first, last = int(week_of(ts.min())) - 2, int(week_of(ts.max()))
    from_mart = popularity_from_mart(source, 123, 60, first, last)
    counter = PopularityCounter(60, first, last)
    empty = np.zeros(0, dtype=np.int64)
    _window_pass(source, 123, counter, empty, empty, empty, datetime(2100, 1, 1))
    from_interactions = counter.table()
    assert np.array_equal(from_mart.positives, from_interactions.positives)
    assert np.array_equal(from_mart.reviews, from_interactions.reviews)
    assert from_mart.reviews[-1].sum() == len(marts["interactions"])
