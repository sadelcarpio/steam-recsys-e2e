import json

import boto3
import pytest
import torch
from moto import mock_aws

from steam_training.artifacts import ArtifactStore
from steam_training.contracts import RecallMetrics, SegmentRecall
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
    # a finished training run deletes its checkpoint
    run_training(settings, source, store)
    assert not any(k.startswith("checkpoints/") for k in _keys(store))


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
    assert not any(k.startswith("checkpoints/") for k in _keys(store))
