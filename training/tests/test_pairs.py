import re

import boto3
import pytest
from moto import mock_aws

from steam_training import launch
from steam_training.artifacts import ArtifactStore
from steam_training.config import PairSettings, RankerSettings
from steam_training.contracts import PairEvaluation, RankingMetrics
from steam_training.pairs import decide_pair, run_pair
from steam_training.pipeline import run_promotion, run_training
from steam_training.ranker.pipeline import metrics_line, run_ranking
from tests.conftest import BUCKET, FakeSource, make_marts


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def _keys(store) -> set[str]:
    listing = store.s3.list_objects_v2(Bucket=BUCKET)
    return {o["Key"] for o in listing.get("Contents", [])}


def metrics(**recall) -> RankingMetrics:
    """recall@k from keyword args like r20=0.1; ndcg = recall / 2."""
    values = {int(k[1:]): v for k, v in recall.items()}
    return RankingMetrics(groups=10, recall=values, ndcg={k: v / 2 for k, v in values.items()})


def evaluation(final: dict, retrieval: dict, ranker="k") -> PairEvaluation:
    return PairEvaluation(
        retriever_id="r", ranker_id=ranker, final=metrics(**final), retrieval=metrics(**retrieval)
    )


SETTINGS = dict(model_artifacts_bucket=BUCKET, ranker_id="k1", pair_top_k=20)


def test_decide_pair():
    s = PairSettings(**SETTINGS, pair_candidates=100, pair_retrieval_k=50)
    champion = evaluation({"r20": 0.10}, {"r50": 0.20, "r100": 0.30})
    better = evaluation({"r20": 0.12}, {"r50": 0.21, "r100": 0.30})
    chosen, _ = decide_pair(champion, [better], retriever_changed=True, settings=s)
    assert chosen is better
    # a new retriever must win at retrieval and keep the ceiling
    weak = evaluation({"r20": 0.12}, {"r50": 0.19, "r100": 0.30})
    assert decide_pair(champion, [weak], True, s)[0] is None
    lower_ceiling = evaluation({"r20": 0.12}, {"r50": 0.21, "r100": 0.29})
    chosen, reason = decide_pair(champion, [lower_ceiling], True, s)
    assert chosen is None and "ceiling" in reason
    # a ranker-only challenge skips the retriever gate; the final list must win
    assert decide_pair(champion, [weak], False, s)[0] is weak
    same = evaluation({"r20": 0.10}, {"r50": 0.21, "r100": 0.30})
    assert decide_pair(champion, [same], False, s)[0] is None
    # the best of several challengers
    chosen, _ = decide_pair(champion, [better, weak], False, s)
    assert chosen is better


def test_pair_settings_validation():
    with pytest.raises(ValueError, match="RETRIEVER_ID"):
        PairSettings(model_artifacts_bucket=BUCKET)
    assert PairSettings(model_artifacts_bucket=BUCKET, rollback=True).rollback
    with pytest.raises(ValueError, match="champion"):
        PairSettings(model_artifacts_bucket=BUCKET, ranker_id="champion")


def test_pair_lifecycle(settings, store):
    source = FakeSource(make_marts(n_users=600, reviews_per_user=10, seed=1))
    retriever = settings.model_copy(update={"holdout_weeks": 16, "test_weeks": 6})
    run_training(retriever, source, store)
    run_promotion(retriever, source, store)  # champion retriever, no pair.json yet
    ranker = RankerSettings(
        model_artifacts_bucket=BUCKET, ranker_id="k1", retriever_id=settings.model_id,
        device="cpu", ranker_candidates=15, ranker_top_k=5, ranker_history_length=5,
        ranker_min_groups=10, ranker_min_data_in_leaf=5, ranker_max_rounds=20,
        ranker_ablation=False, ranker_list_users=0, eval_batch_size=128,
    )  # fmt: skip
    run_ranking(ranker, source, store)
    pair = PairSettings(
        model_artifacts_bucket=BUCKET, ranker_id="k1", device="cpu", pair_candidates=15,
        pair_top_k=5, pair_retrieval_k=10, pair_history_length=5, pair_list_users=100,
        eval_batch_size=128, force_promotion=True,  # synthetic data: gates may go either way
    )  # fmt: skip
    report = run_pair(pair, source, store)
    assert report.promoted and report.chosen.ranker_id == "k1"
    assert report.champion.ranker_id is None and report.champion.retriever_id == "abc123"
    # groups from the ranker's T2 on: neither model trained or stopped early on them
    assert report.window_start == store.load_ranker_metadata("k1").window.t2
    assert report.chosen.user_lists is not None
    served = store.read_pair()
    assert (served.retriever_id, served.ranker_id) == ("abc123", "k1")
    assert {"rankers/champion/model.txt", "rankers/champion/metadata.json"} <= _keys(store)
    assert [r.ranker_id for r in store.read_history()] == [None, "k1"]
    assert "evaluation/pairs/abc123__k1.json" in _keys(store)

    # a retriever-only promotion would leave k1 serving with an unevaluated retriever
    with pytest.raises(ValueError, match="pair"):
        run_promotion(retriever, source, store)

    back = run_pair(PairSettings(model_artifacts_bucket=BUCKET, rollback=True), source, store)
    assert (back.retriever_id, back.ranker_id) == ("abc123", None)
    assert back.reason.startswith("rollback")
    assert not {"rankers/champion/model.txt", "rankers/champion/metadata.json"} & _keys(store)
    assert len(store.read_history()) == 3
    with pytest.raises(ValueError, match="roll back"):
        run_pair(PairSettings(model_artifacts_bucket=BUCKET, rollback=True), source, store)


def test_launcher_modes():
    assert set(launch.MODES) == {"train", "promote", "rank", "pair"}
    line = (
        "retrieval_recall_at_k=0.117130 ranker_recall_at_k=0.152680 "
        "retrieval_ndcg_at_k=0.052573 ranker_ndcg_at_k=0.075799 candidate_recall=0.265670"
    )
    parsed = {
        name: re.search(regex, line).group(1)
        for name, regex in launch.METRIC_DEFINITIONS["rank"].items()
        if re.search(regex, line)
    }
    assert parsed["ranker:recall"] == "0.152680" and parsed["candidates:recall"] == "0.265670"
    assert callable(metrics_line)


def test_ranker_settings_take_the_launchers_model_id(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "abc")
    monkeypatch.setenv("MODEL_ARTIFACTS_BUCKET", BUCKET)
    assert RankerSettings().ranker_id == "abc"
