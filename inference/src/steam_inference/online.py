"""Online bundle, the catalog side: what the serving Lambda needs to score a history of liked
games against the current catalog. The user tower side is the model's own numpy export
(models/<model_id>/user_tower.npz, written by training: steam_training.export).

Written every run, after retrieval (item embeddings depend on the weekly `reviews_ratio` and
new games): catalog.npz (`contracts.ONLINE_BUNDLE_ARRAYS`) first, then manifest.json pinned to
the new catalog's S3 version id and naming the user tower. Skipped (the previous manifest stays)
when the model has no numpy user tower yet. Old catalog versions expire with the bucket's
noncurrent-version lifecycle.
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import boto3
import numpy as np

from steam_inference.contracts import ONLINE_BUNDLE_ARRAYS, OnlineBundleManifest
from steam_inference.features import Games

log = logging.getLogger(__name__)

CATALOG_NAME = "catalog.npz"
MANIFEST_NAME = "manifest.json"


def build_catalog(games: Games, item_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        "item_embeddings": np.ascontiguousarray(item_embeddings, dtype=np.float32),
        "item_game_id": games.game_id.astype(np.int64),
        "item_game_idx": games.catalog.items.game_idx.astype(np.int64),
        **_names(games.name),
    }
    assert tuple(arrays) == ONLINE_BUNDLE_ARRAYS
    return arrays


def _names(names: np.ndarray) -> dict[str, np.ndarray]:
    """Names as UTF-8 bytes + offsets (a string array would need pickle to load)."""
    encoded = [str(name).encode() for name in names]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return {
        "item_name_utf8": np.frombuffer(b"".join(encoded), dtype=np.uint8).copy(),
        "item_name_offsets": offsets,
    }


def encode_catalog(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # floats barely compress: plain npz loads faster
    return buffer.getvalue()


class BundleStore(Protocol):
    def publish(
        self,
        payload: bytes,
        *,
        model_id: str,
        generated_at: datetime,
        catalog_games: int,
        user_tower_key: str,
    ) -> str:
        """Store the catalog, then the manifest; return where the manifest is."""
        ...


def _manifest(payload: bytes, key: str, version_id: str | None, **fields) -> OnlineBundleManifest:
    return OnlineBundleManifest(
        catalog_key=key,
        catalog_version_id=version_id,
        catalog_sha256=hashlib.sha256(payload).hexdigest(),
        **fields,
    )


class S3BundleStore:
    def __init__(self, bucket: str, prefix: str, *, region: str) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", region_name=region)

    def publish(self, payload: bytes, **fields) -> str:
        key = f"{self.prefix}/{CATALOG_NAME}"
        put = self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)
        manifest = _manifest(payload, key, put.get("VersionId"), **fields)
        manifest_key = f"{self.prefix}/{MANIFEST_NAME}"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=manifest.model_dump_json(indent=2).encode(),
            ContentType="application/json",
        )
        uri = f"s3://{self.bucket}/{manifest_key}"
        log.info("online catalog: %.1f MB, manifest %s", len(payload) / 1e6, uri)
        return uri


class LocalBundleStore:
    """Dry runs: the catalog and its manifest in a local directory."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

    def publish(self, payload: bytes, **fields) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / CATALOG_NAME).write_bytes(payload)
        manifest = _manifest(payload, CATALOG_NAME, None, **fields)
        path = self.directory / MANIFEST_NAME
        path.write_text(manifest.model_dump_json(indent=2))
        return str(path)
