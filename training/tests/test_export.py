import io

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws

from steam_training import __main__ as cli
from steam_training.artifacts import ArtifactStore
from steam_training.contracts import (
    OOV_ID,
    PADDING_ID,
    USER_HISTORY_LENGTH,
    EvaluationReport,
    ModelConfig,
    ModelEvaluation,
    VocabSizes,
)
from steam_training.export import USER_TOWER_ARRAYS, USER_TOWER_NUMPY_FORMAT, user_tower_arrays
from steam_training.model import TwoTowerModel
from tests.conftest import BUCKET

CONFIG = ModelConfig(
    vocab=VocabSizes(games=12, developers=4, publishers=4, genres=4, categories=4),
    game_embedding_dim=6,
    attribute_embedding_dim=3,
    hidden_dim=8,
    output_dim=4,
    temperature=0.05,
)


def _model() -> TwoTowerModel:
    torch.manual_seed(0)
    model = TwoTowerModel(CONFIG)
    model.sync_user_table()
    seen = torch.ones(12, dtype=torch.bool)
    seen[5] = False
    model.set_seen_games(seen)
    return model.eval()


def test_arrays_are_the_user_tower():
    model = _model()
    arrays = user_tower_arrays(model, "abc123")
    assert tuple(arrays) == USER_TOWER_ARRAYS
    assert str(arrays["model_id"]) == "abc123"
    assert int(arrays["format_version"]) == USER_TOWER_NUMPY_FORMAT
    assert (int(arrays["history_length"]), int(arrays["padding_id"]), int(arrays["oov_id"])) == (
        USER_HISTORY_LENGTH,
        PADDING_ID,
        OOV_ID,
    )
    assert np.array_equal(arrays["game_table"], model.user_tower.game_table.numpy())
    assert arrays["seen_games"].tolist() == model.user_tower.game_ids.seen_games.tolist()
    # the MLP in numpy reproduces torch (full-history case; OOV / padding: inference parity test)
    history = torch.tensor([[2, 3, 4, 6, 7]])
    with torch.no_grad():
        expected = model.user_tower(history).numpy()[0]
    x = np.concatenate([arrays["game_table"][[2, 3, 4, 6, 7]].mean(axis=0), [1.0]])
    out = np.maximum(x @ arrays["w1"].T + arrays["b1"], 0) @ arrays["w2"].T + arrays["b2"]
    np.testing.assert_allclose(out / np.linalg.norm(out), expected, atol=1e-6)


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def _load_npz(store, key):
    body = store.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    with np.load(io.BytesIO(body), allow_pickle=False) as npz:  # no pickle needed
        return {name: npz[name] for name in npz.files}


def _metadata(model_id):
    from datetime import UTC, datetime

    from steam_training.contracts import ModelMetadata, RecallMetrics, SegmentRecall, SplitInfo

    seg = SegmentRecall(rows=1, recall={5: 0.5})
    metrics = RecallMetrics(warm=seg, cold=seg, all=seg)
    return ModelMetadata(
        model_id=model_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config=CONFIG,
        split=SplitInfo(cutoff=datetime(2026, 1, 1), train_rows=1, validation_rows=1),
        snapshots={},
        hyperparameters={},
        validation=metrics,
        popularity_baseline=metrics,
        epoch_losses=[1.0],
    )


def _report(model_id):
    metadata = _metadata(model_id)
    return EvaluationReport(
        model_id=model_id,
        evaluated_at=metadata.created_at,
        primary_k=5,
        split=metadata.split,
        candidate=ModelEvaluation(model_id=model_id, metrics=metadata.validation),
        champion=None,
        popularity_baseline=metadata.validation,
        promoted=True,
        reason="test",
    )


def test_save_model_writes_the_numpy_tower_and_promotion_copies_it(store):
    store.save_model(_model(), _metadata("abc123"))
    arrays = _load_npz(store, "models/abc123/user_tower.npz")
    assert str(arrays["model_id"]) == "abc123"
    store.promote(_report("abc123"))
    assert str(_load_npz(store, "models/champion/user_tower.npz")["model_id"]) == "abc123"


def test_promoting_a_model_without_export_drops_the_stale_champion_file(store):
    store.save_model(_model(), _metadata("abc123"))
    store.promote(_report("abc123"))
    legacy = _model()
    store.save_model(legacy, _metadata("old999"))
    store.s3.delete_object(Bucket=BUCKET, Key="models/old999/user_tower.npz")  # pre-export model
    store.promote(_report("old999"))
    keys = {o["Key"] for o in store.s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert "models/champion/user_tower.npz" not in keys  # never pair abc123's with old999
    assert store.load_metadata("champion").model_id == "old999"


def test_export_backfills_a_model_and_the_champion(store, monkeypatch):
    store.save_model(_model(), _metadata("legacy"))
    store.promote(_report("legacy"))
    for key in ("models/legacy/user_tower.npz", "models/champion/user_tower.npz"):
        store.s3.delete_object(Bucket=BUCKET, Key=key)
    monkeypatch.delenv("USE_SSM", raising=False)
    monkeypatch.setenv("MODEL_ARTIFACTS_BUCKET", BUCKET)
    monkeypatch.setattr(cli, "ArtifactStore", lambda bucket: store)
    cli.main(["export", "--model-id", "champion"])
    for key in ("models/legacy/user_tower.npz", "models/champion/user_tower.npz"):
        assert str(_load_npz(store, key)["model_id"]) == "legacy"


def test_export_needs_a_model_id(store, monkeypatch):
    monkeypatch.delenv("USE_SSM", raising=False)
    monkeypatch.setenv("MODEL_ARTIFACTS_BUCKET", BUCKET)
    with pytest.raises(SystemExit):
        cli.main(["export"])
