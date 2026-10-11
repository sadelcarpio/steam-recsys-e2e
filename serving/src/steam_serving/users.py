"""Demo user index (spec 13): user_idx (1 = most active user) -> pseudonymous user_id.

Inference publishes it every run in the model artifacts bucket (steam_inference.user_index):
serving/users/index.bin holds the user ids as little-endian int64 in user_idx order (user i at
byte offset (i - 1) * 8), and serving/users/index.json names it, pinned to its S3 version. A
lookup is one ranged GetObject of 8 bytes at that version; the manifest is cached and re-read
at most every `refresh_seconds` (a failed refresh keeps the cached one). user_idx changes every
run, so the response always names the resolved user_id too.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import boto3
from pydantic import BaseModel, ConfigDict, Field

from steam_serving.online import S3_CONFIG

log = logging.getLogger(__name__)

SUPPORTED_USER_INDEX_FORMAT = 1
ENTRY_BYTES = 8


class UserIndexManifest(BaseModel):
    """serving/users/index.json (inference `UserIndexManifest`; tolerant reader)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    format_version: int
    generated_at: datetime
    max_user: int = Field(ge=1)
    index_key: str
    index_version_id: str | None = None


class UserIndex:
    def __init__(self, bucket: str, prefix: str, *, region: str, refresh_seconds: int, s3=None):
        self.bucket = bucket
        self.manifest_key = f"{prefix.strip('/')}/index.json"
        self.refresh_seconds = refresh_seconds
        self.s3 = s3 or boto3.client("s3", region_name=region, config=S3_CONFIG)
        self._manifest: UserIndexManifest | None = None
        self._checked_at = float("-inf")

    def manifest(self) -> UserIndexManifest | None:
        """The current manifest, or None when no index was ever published."""
        now = time.monotonic()
        if self._manifest is not None and now - self._checked_at < self.refresh_seconds:
            return self._manifest
        self._checked_at = now
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.manifest_key)["Body"].read()
            manifest = UserIndexManifest.model_validate_json(body)
            if manifest.format_version != SUPPORTED_USER_INDEX_FORMAT:
                raise ValueError(f"user index format {manifest.format_version} not supported")
            self._manifest = manifest
        except self.s3.exceptions.NoSuchKey:
            log.warning("no user index at s3://%s/%s", self.bucket, self.manifest_key)
        except Exception:
            if self._manifest is None:
                raise
            log.exception("user index refresh failed: keeping %s", self._manifest.generated_at)
        return self._manifest

    def user_id(self, manifest: UserIndexManifest, user_idx: int) -> str:
        """The user id at `user_idx` (1..manifest.max_user) of that manifest's index."""
        if not 1 <= user_idx <= manifest.max_user:
            raise ValueError(f"user_idx {user_idx} is outside 1..{manifest.max_user}")
        start = (user_idx - 1) * ENTRY_BYTES
        request = {
            "Bucket": self.bucket,
            "Key": manifest.index_key,
            "Range": f"bytes={start}-{start + ENTRY_BYTES - 1}",
        }
        if manifest.index_version_id:
            request["VersionId"] = manifest.index_version_id
        entry = self.s3.get_object(**request)["Body"].read()
        if len(entry) != ENTRY_BYTES:
            raise RuntimeError(f"user index entry {user_idx}: {len(entry)} bytes")
        return str(int.from_bytes(entry, "little", signed=True))
