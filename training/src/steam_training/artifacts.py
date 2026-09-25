"""Model artifacts as plain S3 objects (layout in `contracts.py`)."""

from __future__ import annotations

import io
import logging

import boto3
import torch
from botocore.exceptions import ClientError

from steam_training.config import CHAMPION
from steam_training.contracts import EvaluationReport, ModelMetadata
from steam_training.export import encode_user_tower
from steam_training.model import TwoTowerModel

log = logging.getLogger(__name__)

USER_TOWER = "user_tower.pt"
ITEM_TOWER = "item_tower.pt"
# numpy export of the user tower for the serving Lambda (`export.py`)
USER_TOWER_NUMPY = "user_tower.npz"
METADATA = "metadata.json"
METRICS = "metrics.json"
# metadata last: copied / written after the files it describes
MODEL_FILES = (USER_TOWER, ITEM_TOWER, USER_TOWER_NUMPY, METADATA)


def model_prefix(model_id: str) -> str:
    return f"models/{model_id}/"


def metrics_key(model_id: str) -> str:
    return f"evaluation/{model_id}/{METRICS}"


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

    def checkpoints(self, model_id: str) -> S3Checkpoints:
        return S3Checkpoints(self, model_id)

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
        source, target = model_prefix(report.model_id), model_prefix(CHAMPION)
        for name in MODEL_FILES:
            if name == USER_TOWER_NUMPY and not self._exists(source + name):
                # saved before the numpy export existed: backfill with `export`
                log.warning("%s has no %s: online serving needs `export`", source, name)
                self.s3.delete_object(Bucket=self.bucket, Key=target + name)
                continue
            self.s3.copy_object(
                Bucket=self.bucket,
                Key=target + name,
                CopySource={"Bucket": self.bucket, "Key": source + name},
            )
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
