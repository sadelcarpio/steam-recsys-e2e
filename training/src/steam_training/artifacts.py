"""Model artifacts as plain S3 objects (layout in `contracts.py`)."""

from __future__ import annotations

import io
import logging

import boto3
import torch
from botocore.exceptions import ClientError

from steam_training.config import CHAMPION
from steam_training.contracts import (
    EvaluationReport,
    ModelMetadata,
    PairRecord,
    PairReport,
    RankerMetadata,
    RankerReport,
    TrackingRecord,
)
from steam_training.export import encode_user_tower
from steam_training.model import TwoTowerModel

log = logging.getLogger(__name__)

USER_TOWER = "user_tower.pt"
ITEM_TOWER = "item_tower.pt"
# numpy export of the user tower for the serving Lambda (`export.py`)
USER_TOWER_NUMPY = "user_tower.npz"
METADATA = "metadata.json"
# experiment tracking record (spec 18); not part of MODEL_FILES, so promotion doesn't copy it
TRACKING = "tracking.json"
METRICS = "metrics.json"
# metadata last: copied / written after the files it describes
MODEL_FILES = (USER_TOWER, ITEM_TOWER, USER_TOWER_NUMPY, METADATA)


def model_prefix(model_id: str) -> str:
    return f"models/{model_id}/"


def metrics_key(model_id: str) -> str:
    return f"evaluation/{model_id}/{METRICS}"


# rankers (spec 9): rankers/<ranker_id>/{model.txt, metadata.json}
RANKER_MODEL = "model.txt"
RANKER_REPORT = "ranker.json"
RANKER_FILES = (RANKER_MODEL, METADATA)  # metadata last
# the serving pair (spec 9): written last by a pair promotion / rollback
PAIR_KEY = f"evaluation/{CHAMPION}/pair.json"
HISTORY_KEY = f"evaluation/{CHAMPION}/history.jsonl"


def pair_report_key(retriever_id: str, ranker_id: str | None) -> str:
    return f"evaluation/pairs/{retriever_id}__{ranker_id or 'none'}.json"


def ranker_prefix(ranker_id: str) -> str:
    return f"rankers/{ranker_id}/"


def ranker_report_key(ranker_id: str) -> str:
    return f"evaluation/rankers/{ranker_id}/{RANKER_REPORT}"


def checkpoint_key(model_id: str) -> str:
    return f"checkpoints/{model_id}/checkpoint.pt"


class ArtifactStore:
    def __init__(self, bucket: str, s3_client=None) -> None:
        self.bucket = bucket
        self.s3 = s3_client or boto3.client("s3")

    # ---- raw objects ----

    def _put(self, key: str, body: bytes, content_type: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as err:
            if err.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return False
            raise
        return True

    def _get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as err:
            if err.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    # ---- models ----

    def save_model(self, model: TwoTowerModel, metadata: ModelMetadata) -> None:
        prefix = model_prefix(metadata.model_id)
        for name, module in ((USER_TOWER, model.user_tower), (ITEM_TOWER, model.item_tower)):
            buffer = io.BytesIO()
            torch.save({k: v.cpu() for k, v in module.state_dict().items()}, buffer)
            self._put(prefix + name, buffer.getvalue(), "application/octet-stream")
        self.save_user_tower_numpy(model, metadata.model_id)
        # metadata last: its presence marks a complete model
        self._put(
            prefix + METADATA, metadata.model_dump_json(indent=2).encode(), "application/json"
        )
        log.info("saved model to s3://%s/%s", self.bucket, prefix)

    def save_user_tower_numpy(self, model: TwoTowerModel, model_id: str) -> str:
        """Write models/<model_id>/user_tower.npz (also used to backfill older models)."""
        key = model_prefix(model_id) + USER_TOWER_NUMPY
        self._put(key, encode_user_tower(model, model_id), "application/octet-stream")
        return key

    def user_tower_numpy_key(self, model_id: str) -> str:
        return model_prefix(model_id) + USER_TOWER_NUMPY

    def has_user_tower_numpy(self, model_id: str) -> bool:
        return self._exists(self.user_tower_numpy_key(model_id))

    def copy_user_tower_numpy(self, source_model_id: str, target_model_id: str) -> str:
        key = model_prefix(target_model_id) + USER_TOWER_NUMPY
        self.s3.copy_object(
            Bucket=self.bucket,
            Key=key,
            CopySource={
                "Bucket": self.bucket,
                "Key": model_prefix(source_model_id) + USER_TOWER_NUMPY,
            },
        )
        return key

    def load_metadata(self, model_id: str) -> ModelMetadata | None:
        body = self._get(model_prefix(model_id) + METADATA)
        return ModelMetadata.model_validate_json(body) if body is not None else None

    def load_model(self, model_id: str) -> tuple[TwoTowerModel, ModelMetadata]:
        metadata = self.load_metadata(model_id)
        if metadata is None:
            raise FileNotFoundError(f"no model at s3://{self.bucket}/{model_prefix(model_id)}")
        model = TwoTowerModel(metadata.config)
        for name, module in ((USER_TOWER, model.user_tower), (ITEM_TOWER, model.item_tower)):
            body = self._get(model_prefix(model_id) + name)
            if body is None:
                raise FileNotFoundError(f"missing {name} for model {model_id}")
            module.load_state_dict(
                torch.load(io.BytesIO(body), weights_only=True, map_location="cpu")
            )
        model.eval()
        return model, metadata

    def save_tracking(self, record: TrackingRecord) -> str:
        key = model_prefix(record.model_id) + TRACKING
        self._put(key, record.model_dump_json(indent=2).encode(), "application/json")
        return key

    def load_tracking(self, model_id: str) -> TrackingRecord | None:
        body = self._get(model_prefix(model_id) + TRACKING)
        return TrackingRecord.model_validate_json(body) if body is not None else None

    def model_uri(self, model_id: str) -> str:
        return f"s3://{self.bucket}/{model_prefix(model_id)}"

    def checkpoints(self, model_id: str) -> S3Checkpoints:
        return S3Checkpoints(self, model_id)

    # ---- rankers (spec 9) ----

    def save_ranker(self, model_text: str, metadata: RankerMetadata) -> None:
        prefix = ranker_prefix(metadata.ranker_id)
        self._put(prefix + RANKER_MODEL, model_text.encode(), "text/plain")
        # metadata last: its presence marks a complete ranker
        self._put(
            prefix + METADATA, metadata.model_dump_json(indent=2).encode(), "application/json"
        )
        log.info("saved ranker to s3://%s/%s", self.bucket, prefix)

    def load_ranker_metadata(self, ranker_id: str) -> RankerMetadata | None:
        body = self._get(ranker_prefix(ranker_id) + METADATA)
        return RankerMetadata.model_validate_json(body) if body is not None else None

    def load_ranker(self, ranker_id: str) -> tuple[str, RankerMetadata]:
        """(LightGBM model text, metadata) of a saved ranker."""
        metadata = self.load_ranker_metadata(ranker_id)
        body = self._get(ranker_prefix(ranker_id) + RANKER_MODEL)
        if metadata is None or body is None:
            raise FileNotFoundError(f"no ranker at s3://{self.bucket}/{ranker_prefix(ranker_id)}")
        return body.decode(), metadata

    def write_ranker_report(self, report: RankerReport) -> str:
        key = ranker_report_key(report.ranker_id)
        self._put(key, report.model_dump_json(indent=2).encode(), "application/json")
        log.info("wrote s3://%s/%s", self.bucket, key)
        return key

    def read_ranker_report(self, ranker_id: str) -> RankerReport | None:
        body = self._get(ranker_report_key(ranker_id))
        return RankerReport.model_validate_json(body) if body is not None else None

    def copy_ranker_to_champion(self, ranker_id: str | None) -> None:
        """rankers/champion/ = a copy of rankers/<ranker_id>/ (metadata last); None deletes it,
        so a stale ranker never serves with another retriever."""
        target = ranker_prefix(CHAMPION)
        if ranker_id is None:
            for name in reversed(RANKER_FILES):  # metadata first: no half-complete ranker
                self.s3.delete_object(Bucket=self.bucket, Key=target + name)
            return
        source = ranker_prefix(ranker_id)
        for name in RANKER_FILES:
            self.s3.copy_object(
                Bucket=self.bucket,
                Key=target + name,
                CopySource={"Bucket": self.bucket, "Key": source + name},
            )

    # ---- serving pair (spec 9) ----

    def read_pair(self) -> PairRecord | None:
        body = self._get(PAIR_KEY)
        return PairRecord.model_validate_json(body) if body is not None else None

    def read_history(self) -> list[PairRecord]:
        body = self._get(HISTORY_KEY)
        if body is None:
            return []
        return [PairRecord.model_validate_json(line) for line in body.decode().splitlines() if line]

    def write_pair(self, record: PairRecord) -> None:
        """Appends to history.jsonl, then writes pair.json (last: it names what serves)."""
        lines = [r.model_dump_json() for r in self.read_history()] + [record.model_dump_json()]
        self._put(HISTORY_KEY, ("\n".join(lines) + "\n").encode(), "application/x-ndjson")
        self._put(PAIR_KEY, record.model_dump_json(indent=2).encode(), "application/json")
        log.info("serving pair: %s + %s", record.retriever_id, record.ranker_id or "no ranker")

    def write_pair_report(self, report: PairReport) -> str:
        chosen = report.chosen or (report.challengers[0] if report.challengers else None)
        key = pair_report_key(
            chosen.retriever_id if chosen else "none", chosen.ranker_id if chosen else None
        )
        self._put(key, report.model_dump_json(indent=2).encode(), "application/json")
        log.info("wrote s3://%s/%s", self.bucket, key)
        return key

    def copy_model_to_champion(self, model_id: str) -> None:
        """models/champion/ = a copy of models/<model_id>/ (metadata last). A model without the
        numpy user tower deletes the champion's copy."""
        source, target = model_prefix(model_id), model_prefix(CHAMPION)
        for name in MODEL_FILES:
            if name == USER_TOWER_NUMPY and not self._exists(source + name):
                log.warning("%s has no %s: online serving needs `export`", source, name)
                self.s3.delete_object(Bucket=self.bucket, Key=target + name)
                continue
            self.s3.copy_object(
                Bucket=self.bucket,
                Key=target + name,
                CopySource={"Bucket": self.bucket, "Key": source + name},
            )

    # ---- evaluation + promotion ----

    def write_report(self, report: EvaluationReport, model_id: str | None = None) -> None:
        key = metrics_key(model_id or report.model_id)
        self._put(key, report.model_dump_json(indent=2).encode(), "application/json")
        log.info("wrote s3://%s/%s", self.bucket, key)

    def read_report(self, model_id: str) -> EvaluationReport | None:
        body = self._get(metrics_key(model_id))
        return EvaluationReport.model_validate_json(body) if body is not None else None

    def promote(self, report: EvaluationReport) -> None:
        """Copy the candidate to models/champion/ and its report to evaluation/champion/ (last,
        so the champion metrics never describe a model that is not fully in place)."""
        self.copy_model_to_champion(report.model_id)
        self.write_report(report, model_id=CHAMPION)
        log.info("promoted %s to champion", report.model_id)


class S3Checkpoints:
    """Per-epoch training state at checkpoints/<model_id>/checkpoint.pt (`train.Checkpoints`).
    Kept after the model is saved (a finished run can be extended); the bucket lifecycle
    expires it after 14 days."""

    def __init__(self, store: ArtifactStore, model_id: str) -> None:
        self.store = store
        self.key = checkpoint_key(model_id)

    def load(self) -> dict | None:
        body = self.store._get(self.key)
        if body is None:
            return None
        return torch.load(io.BytesIO(body), weights_only=True, map_location="cpu")

    def save(self, state: dict) -> None:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self.store._put(self.key, buffer.getvalue(), "application/octet-stream")
        log.info(
            "checkpoint after epoch %d: s3://%s/%s", state["epoch"], self.store.bucket, self.key
        )
