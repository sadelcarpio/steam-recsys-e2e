import json
import logging
import re

import boto3
import pytest
import torch
from moto import mock_aws

from steam_training.artifacts import ArtifactStore
from steam_training.contracts import RecallMetrics, SegmentRecall
from steam_training.launch import METRIC_DEFINITIONS
from steam_training.pipeline import decide, run_promotion, run_training
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


def _metrics(warm: float) -> RecallMetrics:
    seg = SegmentRecall(rows=10, recall={50: warm})
    return RecallMetrics(warm=seg, cold=seg, all=seg)


def test_decide():
    base = _metrics(0.1)
    assert decide(50, _metrics(0.3), None, base, 0.0)[0]
    assert not decide(50, _metrics(0.05), None, base, 0.0)[0]
    assert decide(50, _metrics(0.3), _metrics(0.2), base, 0.0)[0]
    assert not decide(50, _metrics(0.3), _metrics(0.3), base, 0.0)[0]
    assert not decide(50, _metrics(0.3), _metrics(0.25), base, 0.1)[0]


def test_training_writes_artifacts_that_reload(settings, source, store):
    metadata = run_training(settings, source, store)
    assert _keys(store) == {
        "models/abc123/user_tower.pt",
        "models/abc123/item_tower.pt",
        "models/abc123/user_tower.npz",
        "models/abc123/metadata.json",
        "checkpoints/abc123/checkpoint.pt",
    }
    assert metadata.snapshots["interactions"] == 123
    assert metadata.hyperparameters["epochs"] == settings.epochs
    assert "model_artifacts_bucket" not in metadata.hyperparameters
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, loaded = store.load_model("abc123")
    assert loaded == metadata
    assert torch.equal(model.user_tower.game_table, model.item_tower.game_embedding.weight) is False
    # the user table is the snapshot from the last epoch start, not re-synced after training
    assert model.user_tower.game_ids.seen_games.any()


def test_promotion_flow(settings, source, store):
    run_training(settings, source, store)
    first = run_promotion(settings, source, store)
    assert first.promoted and first.champion is None
    assert {"models/champion/user_tower.pt", "models/champion/user_tower.npz"} <= _keys(store)
    champion_metrics = json.loads(
        store.s3.get_object(Bucket=BUCKET, Key="evaluation/champion/metrics.json")["Body"].read()
    )
    assert champion_metrics["model_id"] == "abc123"
    assert store.load_metadata("champion").model_id == "abc123"

    # re-promoting the champion is a no-op
    again = run_promotion(settings, source, store)
    assert not again.promoted and again.reason == "already the champion"

    # a weaker candidate (1 epoch, tiny model) on the same data is not promoted
    weak = settings.model_copy(
        update={"model_id": "weak1", "epochs": 1, "learning_rate": 1e-6, "mined_negatives": False}
    )
    run_training(weak, source, store)
    report = run_promotion(weak, source, store)
    assert not report.promoted
    assert report.champion is not None and report.champion.model_id == "abc123"
    assert store.load_metadata("champion").model_id == "abc123"
    assert store.read_report("weak1") == report


def test_forced_promotion_swaps_a_losing_candidate(settings, source, store):
    run_training(settings, source, store)
    run_promotion(settings, source, store)
    weak = settings.model_copy(
        update={
            "model_id": "weak1",
            "epochs": 1,
            "learning_rate": 1e-6,
            "mined_negatives": False,
            "force_promotion": True,
        }
    )
    run_training(weak, source, store)
    report = run_promotion(weak, source, store)
    assert report.promoted
    assert report.reason.startswith("forced (FORCE_PROMOTION): ")
    assert "does not beat" in report.reason  # why it would have been rejected
    # the real metrics are kept: the champion it replaced is still in the report
    assert report.champion is not None and report.champion.model_id == "abc123"
    assert store.load_metadata("champion").model_id == "weak1"
    assert store.read_report("champion") == report


def test_promotion_evaluates_after_both_cutoffs(settings, store):
    old = FakeSource(make_marts(n_users=300, seed=0))
    run_training(settings, old, store)
    run_promotion(settings, old, store)
    # more (later) data: the new model's cutoff is later than the champion's
    newer = make_marts(n_users=300, seed=0)
    newer_source = FakeSource(newer)
    candidate = settings.model_copy(update={"model_id": "def456", "validation_fraction": 0.05})
    meta = run_training(candidate, newer_source, store)
    report = run_promotion(candidate, newer_source, store)
    champion_cutoff = store.load_metadata("champion").split.cutoff
    assert report.split.cutoff == max(meta.split.cutoff, champion_cutoff)


def _parsed(mode: str, caplog) -> dict[str, list[float]]:
    """What SageMaker would extract from the job's log lines with the launcher's definitions."""
    lines = [record.getMessage() for record in caplog.records]
    return {
        name: [float(m.group(1)) for line in lines if (m := re.search(regex, line))]
        for name, regex in METRIC_DEFINITIONS[mode].items()
    }


def _logged(values, tolerance: float = 1e-6):
    return pytest.approx(values, abs=tolerance)  # the log lines are rounded


def test_metric_definitions_parse_the_job_logs(settings, source, store, caplog):
    caplog.set_level(logging.INFO)
    settings = settings.model_copy(update={"epoch_eval_rows": 100})  # per-epoch recall line
    metadata = run_training(settings, source, store)
    train = _parsed("train", caplog)
    assert train["train:loss"] == _logged(metadata.epoch_losses, 1e-4)
    assert len(train["train:monitor_warm_recall"]) == settings.epochs
    k = settings.primary_k
    assert train["final:warm_recall"] == _logged([metadata.validation.warm.recall[k]])
    assert train["final:all_recall"] == _logged([metadata.validation.all.recall[k]])
    baseline = metadata.popularity_baseline
    assert train["popularity:warm_recall"] == _logged([baseline.warm.recall[k]])
    assert train["popularity:all_recall"] == _logged([baseline.all.recall[k]])

    run_promotion(settings, source, store)  # first model: no champion line
    candidate = settings.model_copy(update={"model_id": "def456", "epochs": 1})
    run_training(candidate, source, store)
    caplog.clear()
    report = run_promotion(candidate, source, store)
    promote = _parsed("promote", caplog)
    assert promote["candidate:warm_recall"] == _logged([report.primary(report.candidate.metrics)])
    assert promote["champion:warm_recall"] == _logged([report.primary(report.champion.metrics)])
    assert promote["popularity:warm_recall"] == _logged(
        [report.primary(report.popularity_baseline)]
    )


def test_promotion_requires_a_trained_model(settings, source, store):
    with pytest.raises(FileNotFoundError):
        run_promotion(settings, source, store)


def test_s3_checkpoints_roundtrip_and_cleanup(settings, source, store):
    checkpoints = store.checkpoints("abc123")
    assert checkpoints.load() is None
    checkpoints.save({"epoch": 1, "fingerprint": "f", "mined": torch.arange(3)})
    assert "checkpoints/abc123/checkpoint.pt" in _keys(store)
    loaded = checkpoints.load()
    assert loaded["epoch"] == 1 and torch.equal(loaded["mined"], torch.arange(3))
    # a finished training run keeps its checkpoint (so it can be extended)
    run_training(settings, source, store)
    assert "checkpoints/abc123/checkpoint.pt" in _keys(store)
    assert store.checkpoints("abc123").load()["epoch"] == settings.epochs


def test_training_resumes_from_an_s3_checkpoint(settings, source, store, monkeypatch):
    calls = {"n": 0}
    original = store.checkpoints

    def crashing(model_id):
        cp = original(model_id)
        save = cp.save

        def save_then_crash(state):
            save(state)
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt

        cp.save = save_then_crash
        return cp

    monkeypatch.setattr(store, "checkpoints", crashing)
    with pytest.raises(KeyboardInterrupt):
        run_training(settings, source, store)
    assert "checkpoints/abc123/checkpoint.pt" in _keys(store)
    monkeypatch.setattr(store, "checkpoints", original)
    metadata = run_training(settings, source, store)
    assert len(metadata.epoch_losses) == settings.epochs
