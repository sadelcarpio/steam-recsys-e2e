"""Experiment tracking (spec 18): diagnostics, the tracking record, live MLflow logging, replay
and champion / challenger aliases (MLflow on a local sqlite store)."""

from collections import Counter

import boto3
import numpy as np
import pyarrow.compute as pc
import pytest
import torch
from moto import mock_aws

from steam_training import tracking
from steam_training.__main__ import track
from steam_training.artifacts import ArtifactStore
from steam_training.config import TrackSettings
from steam_training.data import EvalRows, load_training_data
from steam_training.diagnostics import (
    activity_buckets,
    embedding_health,
    gini,
    retrieval_diagnostics,
    user_hit_rate,
)
from steam_training.evaluation import NOT_FOUND, CatalogIndex, Exposure, target_ranks
from steam_training.pipeline import run_training
from tests.conftest import BUCKET

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


@pytest.fixture
def mlflow_uri(tmp_path, monkeypatch):
    # sqlite store; artifacts go to ./mlruns under the temporary working directory
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


@pytest.fixture
def patch_store(monkeypatch, store):
    """`track` builds its own ArtifactStore: hand it the moto-backed one."""
    monkeypatch.setattr("steam_training.__main__.ArtifactStore", lambda bucket: store)
    return store


# ---- diagnostics ----


def test_user_hit_rate_weighs_users_equally():
    ranks = np.array([0, NOT_FOUND, NOT_FOUND, NOT_FOUND, 3])
    users = np.array([1, 1, 1, 1, 2])
    hit_rate, recall, n = user_hit_rate(ranks, users, [1, 5])
    assert n == 2
    assert hit_rate == {1: 0.5, 5: 1.0}  # user 1 hit at 1; user 2 only within 5
    assert recall[1] == pytest.approx((1 / 4 + 0) / 2)
    assert recall[5] == pytest.approx((1 / 4 + 1) / 2)


def test_activity_buckets_split_new_and_returning_users():
    activity = np.array([0, 0, *range(1, 101)])
    users = np.arange(len(activity))
    ranks = np.where(activity >= 50, 0, NOT_FOUND)  # only active users hit
    buckets = {b.name: b for b in activity_buckets(ranks, users, activity, [1])}
    assert list(buckets) == ["new", "p0-50", "p50-80", "p80-95", "p95-99", "p99-100"]
    assert buckets["new"].rows == 2 and buckets["new"].recall[1] == 0
    assert buckets["p0-50"].max_activity == 50 and buckets["p0-50"].recall[1] < 0.1
    assert buckets["p99-100"].recall[1] == 1.0
    assert sum(b.rows for b in buckets.values()) == len(activity)


def test_gini():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.array([0] * 99 + [5])) == pytest.approx(0.99)
    assert gini(np.zeros(3)) == 0.0


def test_embedding_health_detects_collapse():
    rng = torch.Generator().manual_seed(0)
    healthy = embedding_health(torch.randn(2000, 16, generator=rng))
    assert abs(healthy.mean_cosine) < 0.05 and healthy.effective_rank_ratio > 0.8
    base = torch.randn(1, 16, generator=rng)
    collapsed = embedding_health(base + 1e-3 * torch.randn(2000, 16, generator=rng))
    assert collapsed.mean_cosine > 0.99


def test_exposure_counts_what_rows_retrieved():
    index = CatalogIndex(torch.arange(2, 7), torch.tensor([-1, -1, 0, 1, 2, 3, 4]))

    def scores(history):  # catalog rows 4 > 3 > 2 > 1 > 0 for everyone
        return torch.arange(5, dtype=torch.float32).repeat(len(history), 1)

    exposure = Exposure.empty(index)
    history = torch.tensor([[0] * 5, [6, 0, 0, 0, 0]])  # row 2 excludes game 6 (catalog row 4)
    ranks = target_ranks(scores, history, torch.tensor([6, 4]), index, 2, 1, exposure)
    assert ranks.tolist() == [0, 1]
    assert exposure.best_rank.tolist() == [NOT_FOUND, NOT_FOUND, 1, 0, 0]
    assert exposure.count.tolist() == [0, 0, 1, 2, 1]
    rows = EvalRows(
        history.numpy(),
        np.array([6, 4]),
        np.array([False, True]),
        user_id=np.array([1, 2]),
        activity=np.array([0, 3]),
    )
    d = retrieval_diagnostics(ranks, exposure, rows, [1, 2])
    assert d.coverage == {1: 0.4, 2: 0.6}
    assert d.user_hit_rate == {1: 0.5, 2: 1.0}
    assert [b.name for b in d.by_activity] == ["new", "p0-50"]


# ---- data ----


def test_validation_rows_carry_user_activity(source, marts):
    data = load_training_data(
        source, validation_fraction=0.2, cold_row_fraction=0.0, eval_max_rows=10**9, seed=0
    )
    interactions = marts["interactions"]
    before = pc.less(interactions["timestamp"], data.split.cutoff)
    expected = Counter(interactions.filter(before)["user_id"].to_pylist())
    rows = data.validation
    assert rows.user_id is not None and rows.activity is not None
    assert [expected.get(u, 0) for u in rows.user_id.tolist()] == rows.activity.tolist()
    sampled = rows.sample(5, seed=1)
    assert sampled.activity is not None and len(sampled.activity) == 5
    # every game the user reviewed (any polarity) strictly before the row
    users = interactions["user_id"].to_numpy()
    games = interactions["game_idx"].to_numpy()
    seconds = interactions["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    assert rows.reviewed is not None and sampled.reviewed is not None
    for i in range(0, len(rows), 7):
        mine = (users == rows.user_id[i]) & (seconds < rows.timestamp[i])
        assert sorted(rows.reviewed.row(i)) == sorted(games[mine].tolist())
        assert rows.target[i] not in rows.reviewed.row(i)
    heavy = int(np.argmax(np.diff(rows.reviewed.offsets)))
    assert len(rows.reviewed.row(heavy)) > (rows.history[heavy] > 0).sum()


def test_reviewed_before_slices_each_users_past():
    from steam_training.data import reviewed_before

    # user 0 reviewed games 10, 11, 12 at t=1, 2, 3; user 1 reviewed 20 at t=1
    reviewed = reviewed_before(
        user=np.array([0, 1, 0, 0]),
        timestamp=np.array([3, 1, 1, 2]),
        game=np.array([12, 20, 10, 11]),
        row_user=np.array([0, 0, 1, 0]),
        row_timestamp=np.array([3, 1, 5, 9]),
        row_target=np.array([99, 99, 99, 11]),
    )
    assert [reviewed.row(i) for i in range(4)] == [[10, 11], [], [20], [10, 12]]


def test_evaluation_excludes_every_reviewed_game():
    from steam_training.data import Ragged
    from steam_training.evaluation import rank_rows

    index_games = np.arange(2, 7)  # catalog = games 2..6, scored 6 > 5 > 4 > 3 > 2

    class _Catalog:
        row_of = np.array([-1, -1, 0, 1, 2, 3, 4])
        items = type("Items", (), {"game_idx": index_games})()

        def __len__(self):
            return 5

    def scores(history):
        return torch.arange(5, dtype=torch.float32).repeat(len(history), 1)

    rows = EvalRows(
        np.zeros((2, 5), np.int64),
        np.array([4, 4]),
        np.array([True, True]),
        reviewed=Ragged.from_lists([[], [6, 5]]),  # row 2 already reviewed the top two
    )
    ranks, _ = rank_rows(scores, rows, _Catalog(), 5, 1)
    assert ranks.tolist() == [2, 0]


# ---- lineage ----


def test_lineage_tags(settings):
    from steam_training.contracts import ModelConfig, VocabSizes

    vocab = VocabSizes(games=10, developers=3, publishers=3, genres=3, categories=3, tags=4)
    config = ModelConfig(
        vocab=vocab,
        game_embedding_dim=8,
        attribute_embedding_dim=2,
        hidden_dim=8,
        output_dim=8,
        temperature=0.05,
    )
    tags = tracking.lineage_tags("m1", "abc", config, snapshots={"interactions": 7})
    assert tags["git_commit"] == "abc" and tags["snapshot.interactions"] == "7"
    assert "tags" in tags["features.item"].split(",")
    no_tags = config.model_copy(update={"vocab": vocab.model_copy(update={"tags": None})})
    assert tracking.feature_signature(no_tags) != tags["feature_signature"]
    assert tracking.params_of({"recall_ks": [5, 10], "lr": 0.1}) == {
        "lr": "0.1",
        "recall_ks": "[5, 10]",
    }


# ---- runs ----


def _run(client: MlflowClient, model_id: str):
    experiment = client.get_experiment_by_name("steam-two-tower")
    runs = client.search_runs([experiment.experiment_id], f"tags.model_id = '{model_id}'")
    assert len(runs) == 1
    return runs[0]


def test_training_logs_a_live_mlflow_run(settings, source, store, mlflow_uri):
    settings = settings.model_copy(update={"mlflow_tracking_uri": mlflow_uri, "git_commit": "c0"})
    run_training(settings, source, store)
    client = MlflowClient(mlflow_uri)
    run = _run(client, settings.model_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["git_commit"] == "c0"
    assert run.data.tags["snapshot.interactions"] == "123"
    assert "feature_signature" in run.data.tags and "data.cutoff" in run.data.tags
    assert run.data.params["epochs"] == str(settings.epochs)
    metrics = run.data.metrics
    for key in (
        "validation/warm_recall_at_5",
        "validation/user_hit_rate_at_5",
        "validation/coverage_at_5",
        "popularity/warm_recall_at_5",
        "lift/warm_recall_at_5",
        "embeddings/item_mean_cosine",
        "embeddings/user_effective_rank_ratio",
    ):
        assert key in metrics, key
    assert metrics["lift/warm_recall_at_5"] > 1
    losses = client.get_metric_history(run.info.run_id, "epoch/loss")
    assert [m.step for m in losses] == [1, 2, 3, 4]
    by_k = client.get_metric_history(run.info.run_id, "curves/validation_warm_recall_by_k")
    assert sorted(m.step for m in by_k) == [5, 10]
    record = store.load_tracking(settings.model_id)
    assert record is not None and len(record.epochs) == 4
    assert record.model.recall.warm.recall[5] == metrics["validation/warm_recall_at_5"]


def test_refit_is_logged_as_a_test_evaluation(settings, source, store, mlflow_uri, caplog):
    settings = settings.model_copy(
        update={
            "mlflow_tracking_uri": mlflow_uri,
            "test_fraction": 0.1,
            "fit_on": "train+validation",
            "epoch_eval_rows": 100,
            "epochs": 1,
        }
    )
    run_training(settings, source, store)
    run = _run(MlflowClient(mlflow_uri), settings.model_id)
    assert run.data.tags["data.evaluation"] == "test"
    assert "test/warm_recall_at_5" in run.data.metrics
    assert not any(k.startswith("validation/") for k in run.data.metrics)
    # the test window is scored once, never per epoch
    assert "epoch/monitor_warm_recall_at_5" not in run.data.metrics
    assert store.load_tracking(settings.model_id).evaluation == "test"


def test_unreachable_server_does_not_fail_training(settings, source, store, caplog):
    settings = settings.model_copy(
        update={"mlflow_tracking_uri": "http://127.0.0.1:9", "epochs": 1}
    )
    import mlflow.environment_variables as env

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(env.MLFLOW_HTTP_REQUEST_MAX_RETRIES.name, "0")
        mp.setenv(env.MLFLOW_HTTP_REQUEST_TIMEOUT.name, "2")
        metadata = run_training(settings, source, store)
    assert metadata.model_id == settings.model_id
    assert store.load_tracking(settings.model_id) is not None
    assert "mlflow start failed" in caplog.text


def test_track_replays_once_and_moves_aliases(settings, source, patch_store, mlflow_uri):
    store = patch_store
    run_training(settings, source, store)  # no tracking server: tracking.json only
    second = settings.model_copy(update={"model_id": "def456", "epochs": 1})
    run_training(second, source, store)
    track_settings = TrackSettings(model_artifacts_bucket=BUCKET, mlflow_tracking_uri=mlflow_uri)

    first_run = track(track_settings, "abc123", "champion", replace=False)
    assert track(track_settings, "abc123", None, replace=False) == first_run  # no duplicate
    client = MlflowClient(mlflow_uri, registry_uri=mlflow_uri)
    replayed = _run(client, "abc123")
    assert replayed.data.metrics["validation/warm_recall_at_5"] > 0
    assert len(client.get_metric_history(first_run, "epoch/loss")) == settings.epochs

    champion = client.get_model_version_by_alias(tracking.MODEL_NAME, "champion")
    assert champion.run_id == first_run
    assert champion.source == f"s3://{BUCKET}/models/abc123/"

    second_run = track(track_settings, "def456", "challenger", replace=False)
    assert client.get_model_version_by_alias(tracking.MODEL_NAME, "challenger").run_id == (
        second_run
    )
    # the challenger wins: champion moves to its (already registered) version
    track(track_settings, "def456", "champion", replace=False)
    moved = client.get_model_version_by_alias(tracking.MODEL_NAME, "champion")
    assert moved.run_id == second_run
    assert len(client.search_model_versions(f"name = '{tracking.MODEL_NAME}'")) == 2


def test_track_without_record_explains(settings, patch_store, mlflow_uri):
    track_settings = TrackSettings(model_artifacts_bucket=BUCKET, mlflow_tracking_uri=mlflow_uri)
    with pytest.raises(FileNotFoundError, match="tracking.json"):
        track(track_settings, "nope", None, replace=False)
