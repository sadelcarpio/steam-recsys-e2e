"""Experiment tracking (spec 18): every training run becomes a `TrackingRecord`
(models/<model_id>/tracking.json) and, when `MLFLOW_TRACKING_URI` is set, an MLflow run that is
logged live (lineage tags and params at start, per-epoch series, final metrics at the end).

`python -m steam_training track --model-id <id>` replays a record into MLflow (runs trained
without a server, e.g. SageMaker jobs) and `--alias champion|challenger` points a registered
model alias at the run's model (`MODEL_NAME`, source = the model's S3 prefix).

Tracking never fails training: an unreachable server is logged once and the run goes on, with
the record still saved to S3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from steam_training.contracts import (
    ARCHITECTURE_VERSION,
    USER_HISTORY_LENGTH,
    EpochMetrics,
    ModelConfig,
    RetrievalDiagnostics,
    SplitInfo,
    TrackingRecord,
)

log = logging.getLogger(__name__)

MODEL_NAME = "steam-two-tower"
ALIASES = ("champion", "challenger")
# MLflow log_batch limits per call
_MAX_METRICS, _MAX_PARAMS, _MAX_TAGS = 1000, 100, 100


# ---- lineage ----


def features(config: ModelConfig) -> dict[str, list[str]]:
    """Inputs of each tower, as the model reads them."""
    item = ["game_idx", "developers", "publishers", "genres", "categories"]
    if config.vocab.tags is not None:
        item.append("tags")
    item += ["is_free", "is_free_known", "reviews_ratio"]
    user = [f"games_reviewed_positive[{USER_HISTORY_LENGTH}]", "history_length"]
    return {"user": user, "item": item}


def feature_signature(config: ModelConfig) -> str:
    """Short hash of the feature lists + architecture: runs with the same signature differ only
    in data or hyperparameters."""
    payload = {
        "features": features(config),
        "architecture_version": config.architecture_version,
        "dims": [config.game_embedding_dim, config.attribute_embedding_dim, config.hidden_dim],
        "output_dim": config.output_dim,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def git_commit(cwd: Path | None = None) -> tuple[str, bool | None]:
    """(HEAD sha, has uncommitted changes) of the checkout at `cwd`; ("unknown", None) outside
    one (e.g. in the image, where the launcher passes GIT_COMMIT)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return sha, bool(status.strip())


def lineage_tags(
    model_id: str,
    commit: str | None,
    config: ModelConfig | None = None,
    split: SplitInfo | None = None,
    snapshots: dict[str, int | None] | None = None,
    run_fingerprint: str | None = None,
) -> dict[str, str]:
    tags = {"model_id": model_id, "architecture_version": str(ARCHITECTURE_VERSION)}
    if commit is None:
        sha, dirty = git_commit()
        tags["git_commit"] = sha
        if dirty is not None:
            tags["git_dirty"] = str(dirty).lower()
    else:
        tags["git_commit"] = commit
    if config is not None:
        tags["feature_signature"] = feature_signature(config)
        tower_features = features(config)
        tags["features.user"] = ",".join(tower_features["user"])
        tags["features.item"] = ",".join(tower_features["item"])
    if split is not None:
        tags["data.cutoff"] = split.cutoff.isoformat()
        tags["data.evaluation"] = split.evaluation
        if split.end is not None:
            tags["data.end"] = split.end.isoformat()
        tags["data.train_rows"] = str(split.train_rows)
        tags["data.validation_rows"] = str(split.validation_rows)
    for table, snapshot in (snapshots or {}).items():
        tags[f"snapshot.{table}"] = str(snapshot)
    if run_fingerprint:
        tags["fingerprint"] = run_fingerprint
    return tags


def params_of(hyperparameters: dict[str, object]) -> dict[str, str]:
    return {
        name: json.dumps(value) if isinstance(value, list | dict) else str(value)
        for name, value in sorted(hyperparameters.items())
    }


# ---- metrics ----


def _diagnostic_metrics(prefix: str, d: RetrievalDiagnostics) -> dict[str, float]:
    out: dict[str, float] = {}
    for segment in ("warm", "cold", "all"):
        recall = getattr(d.recall, segment)
        for k, value in recall.recall.items():
            out[f"{prefix}{segment}_recall_at_{k}"] = value
        out[f"{prefix}{segment}_rows"] = float(recall.rows)
    for k, value in d.user_hit_rate.items():
        out[f"{prefix}user_hit_rate_at_{k}"] = value
    for k, value in d.user_recall.items():
        out[f"{prefix}user_recall_at_{k}"] = value
    for k, value in d.coverage.items():
        out[f"{prefix}coverage_at_{k}"] = value
    out[f"{prefix}exposure_gini"] = d.exposure_gini
    for bucket in d.by_activity:
        for k, value in bucket.recall.items():
            out[f"{prefix}activity/{bucket.name}_recall_at_{k}"] = value
    return out


def final_metrics(record: TrackingRecord) -> dict[str, float]:
    """Flat MLflow metrics of a record: `validation/...` or `test/...` (the model, named after
    the window it was measured on), `popularity/...`, `lift/...` (model / popularity),
    `embeddings/...`."""
    out = _diagnostic_metrics(f"{record.evaluation}/", record.model)
    out |= _diagnostic_metrics("popularity/", record.popularity)
    for segment in ("warm", "all"):
        model = getattr(record.model.recall, segment).recall
        baseline = getattr(record.popularity.recall, segment).recall
        for k, value in model.items():
            if baseline.get(k):
                out[f"lift/{segment}_recall_at_{k}"] = value / baseline[k]
    for name, health in (("item", record.item_embeddings), ("user", record.user_embeddings)):
        out[f"embeddings/{name}_mean_cosine"] = health.mean_cosine
        out[f"embeddings/{name}_effective_rank"] = health.effective_rank
        out[f"embeddings/{name}_effective_rank_ratio"] = health.effective_rank_ratio
    return out


def series_metrics(record: TrackingRecord) -> list[tuple[str, float, int]]:
    """Curves for the MLflow charts, as (key, value, step): recall vs K (step = K) and recall by
    activity bucket at the primary K (step = bucket ordinal, `new` = 0)."""
    out = []
    for segment in ("warm", "all"):
        for k, value in getattr(record.model.recall, segment).recall.items():
            out.append((f"curves/{record.evaluation}_{segment}_recall_by_k", value, k))
    for step, bucket in enumerate(record.model.by_activity):
        if record.primary_k in bucket.recall:
            out.append(
                (
                    f"curves/{record.evaluation}_recall_at_{record.primary_k}_by_activity",
                    bucket.recall[record.primary_k],
                    step,
                )
            )
    return out


def epoch_metrics(m: EpochMetrics, primary_k: int) -> dict[str, float]:
    out = {"epoch/loss": m.loss}
    if m.monitor_warm_recall is not None:
        out[f"epoch/monitor_warm_recall_at_{primary_k}"] = m.monitor_warm_recall
    if m.seconds is not None:
        out["epoch/seconds"] = m.seconds
    return out


# ---- MLflow ----


def _ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


def _chunks[T](items: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MlflowSink:
    """The MLflow side of tracking (one experiment)."""

    def __init__(self, tracking_uri: str, experiment: str, client=None) -> None:
        # no usage telemetry to MLflow's servers unless explicitly turned back on
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        from mlflow.tracking import MlflowClient

        self.client = client or MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._experiment_id: str | None = None

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            found = self.client.get_experiment_by_name(self.experiment)
            self._experiment_id = (
                found.experiment_id if found else self.client.create_experiment(self.experiment)
            )
        return self._experiment_id

    def find_run(self, model_id: str) -> str | None:
        runs = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.model_id = '{model_id}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else None

    def start(self, model_id: str, tags: dict[str, str], started_at: datetime) -> str:
        run = self.client.create_run(
            self.experiment_id, start_time=_ms(started_at), tags=tags, run_name=model_id
        )
        return run.info.run_id

    def log(
        self,
        run_id: str,
        *,
        metrics: Iterable[tuple[str, float, int]] = (),
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        when: datetime | None = None,
    ) -> None:
        from mlflow.entities import Metric, Param, RunTag

        timestamp = _ms(when or datetime.now(UTC))
        metric_list = [Metric(k, float(v), timestamp, step) for k, v, step in metrics]
        for chunk in _chunks(metric_list, _MAX_METRICS):
            self.client.log_batch(run_id, metrics=chunk)
        for chunk in _chunks([Param(k, v) for k, v in (params or {}).items()], _MAX_PARAMS):
            self.client.log_batch(run_id, params=chunk)
        for chunk in _chunks([RunTag(k, v) for k, v in (tags or {}).items()], _MAX_TAGS):
            self.client.log_batch(run_id, tags=chunk)

    def log_epoch(self, run_id: str, m: EpochMetrics, primary_k: int) -> None:
        self.log(run_id, metrics=[(k, v, m.epoch) for k, v in epoch_metrics(m, primary_k).items()])

    def finish(self, run_id: str, record: TrackingRecord, status: str = "FINISHED") -> None:
        self.log(
            run_id,
            metrics=[(k, v, 0) for k, v in final_metrics(record).items()] + series_metrics(record),
            tags=record.tags,
            when=record.ended_at,
        )
        self.client.log_dict(run_id, record.model_dump(mode="json"), "tracking.json")
        self.client.set_terminated(run_id, status=status, end_time=_ms(record.ended_at))

    def fail(self, run_id: str) -> None:
        self.client.set_terminated(run_id, status="FAILED")

    def replay(self, record: TrackingRecord) -> str:
        """A finished run from a record (same content as a live run)."""
        run_id = self.start(record.model_id, record.tags, record.started_at)
        self.log(run_id, params=record.params, when=record.started_at)
        for m in record.epochs:
            self.log_epoch(run_id, m, record.primary_k)
        self.finish(run_id, record)
        return run_id

    def set_alias(self, run_id: str, model_id: str, alias: str, source: str) -> str:
        """Point `MODEL_NAME@alias` at the run's model (registered once per run); returns the
        model version. The alias moves off whatever version held it before."""
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(MODEL_NAME)
        except MlflowException:
            self.client.create_registered_model(
                MODEL_NAME, description="Two-tower retrieval model (training/)"
            )
        versions = self.client.search_model_versions(
            f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
        )
        if versions:
            version = versions[0].version
        else:
            version = self.client.create_model_version(
                MODEL_NAME, source=source, run_id=run_id, tags={"model_id": model_id}
            ).version
        self.client.set_registered_model_alias(MODEL_NAME, alias, version)
        self.client.set_tag(run_id, f"alias.{alias}", "true")
        return str(version)


class RunTracker:
    """Collects a run's record and mirrors it into MLflow when a sink is given. Sink errors are
    logged and turn the sink off (training goes on)."""

    def __init__(self, model_id: str, primary_k: int, sink: MlflowSink | None = None) -> None:
        self.model_id = model_id
        self.primary_k = primary_k
        self.sink = sink
        self.run_id: str | None = None
        self.started_at = datetime.now(UTC)
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.epochs: list[EpochMetrics] = []
        self._started = time.monotonic()

    @classmethod
    def from_settings(cls, settings) -> RunTracker:
        sink = None
        if settings.mlflow_tracking_uri:
            try:
                sink = MlflowSink(settings.mlflow_tracking_uri, settings.mlflow_experiment)
            except ImportError:
                log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed")
        return cls(settings.model_id, settings.primary_k, sink)

    def _call(self, what: str, fn, *args, **kwargs):
        if self.sink is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001 - tracking must never fail training
            log.warning("mlflow %s failed, tracking only to tracking.json: %s", what, err)
            self.sink = None
            return None

    def start(self, tags: dict[str, str], params: dict[str, str]) -> None:
        self.tags |= tags
        self.params |= params
        if self.sink is None:
            return
        sink = self.sink
        self.run_id = self._call("start", sink.start, self.model_id, self.tags, self.started_at)
        if self.run_id is not None:
            self._call("params", sink.log, self.run_id, params=self.params)
            log.info("mlflow run %s (%s)", self.run_id, sink.tracking_uri)

    def tag(self, tags: dict[str, str]) -> None:
        self.tags |= tags
        if self.sink is not None and self.run_id is not None:
            self._call("tags", self.sink.log, self.run_id, tags=tags)

    def epoch(self, m: EpochMetrics) -> None:
        self.epochs.append(m)
        if self.sink is not None and self.run_id is not None:
            self._call("epoch", self.sink.log_epoch, self.run_id, m, self.primary_k)

    def record(self, **results) -> TrackingRecord:
        return TrackingRecord(
            model_id=self.model_id,
            started_at=self.started_at,
            ended_at=datetime.now(UTC),
            primary_k=self.primary_k,
            tags=self.tags,
            params=self.params,
            epochs=self.epochs,
            **results,
        )

    def finish(self, record: TrackingRecord) -> None:
        if self.sink is not None and self.run_id is not None:
            self._call("finish", self.sink.finish, self.run_id, record)

    def fail(self) -> None:
        if self.sink is not None and self.run_id is not None:
            self._call("fail", self.sink.fail, self.run_id)
