"""Online recommendations: the two-tower user tower in numpy, scored against the current
catalog. Two artifacts, both in the model artifacts bucket:
  - the model's user tower, models/<model_id>/user_tower.npz (training: steam_training.export);
  - the catalog, serving/online/catalog.npz (inference: steam_inference.online, every run),
named by serving/online/manifest.json, pinned to the catalog's S3 version.

User tower (training/src/steam_training/model.py, `UserTower`), for one history of game_idx,
most recent first, padded with padding_id:
    ids     -> oov_id when beyond the vocabulary or not seen in training (padding stays padding)
    pooled  = mean of game_table[ids] over the non-padding entries (zeros when empty)
    x       = [pooled, length / history_length]
    user    = normalize(relu(x @ w1.T + b1) @ w2.T + b2)
Scores are cosine similarities with the (already normalized) item embeddings. The inference
test `test_online_parity.py` checks this against torch and batch retrieval.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import boto3
import numpy as np
from botocore.config import Config
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

SUPPORTED_MANIFEST_FORMAT = 2
SUPPORTED_USER_TOWER_FORMAT = 1
EPS = 1e-12  # torch.nn.functional.normalize
CATALOG_ARRAYS = (
    "item_embeddings",
    "item_game_id",
    "item_game_idx",
    "item_name_utf8",
    "item_name_offsets",
)
USER_TOWER_ARRAYS = (
    "format_version",
    "model_id",
    "history_length",
    "padding_id",
    "oov_id",
    "game_table",
    "seen_games",
    "w1",
    "b1",
    "w2",
    "b2",
)


class BundleManifest(BaseModel):
    """serving/online/manifest.json (inference `OnlineBundleManifest`; tolerant reader)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    format_version: int
    model_id: str
    generated_at: datetime
    catalog_key: str
    catalog_version_id: str | None = None
    catalog_sha256: str
    user_tower_key: str


@dataclass(frozen=True)
class OnlineResult:
    game_ids: list[int]  # best first
    names: list[str]
    scores: list[float]  # cosine similarity
    used_game_ids: list[int]  # liked games that made up the history (at most history_length)
    ignored_game_ids: list[int]  # liked games the model does not know (not in the catalog)


@dataclass(frozen=True)
class UserTower:
    model_id: str
    history_length: int
    padding_id: int
    oov_id: int
    game_table: np.ndarray  # float32 [vocab, game_embedding_dim]
    seen_games: np.ndarray  # bool [vocab]
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def from_bytes(cls, payload: bytes) -> UserTower:
        arrays = _load_npz(payload, USER_TOWER_ARRAYS, "user tower")
        if int(arrays["format_version"]) != SUPPORTED_USER_TOWER_FORMAT:
            raise ValueError(
                f"user tower format {int(arrays['format_version'])}, "
                f"this code reads {SUPPORTED_USER_TOWER_FORMAT}"
            )
        return cls(
            model_id=str(arrays["model_id"]),
            history_length=int(arrays["history_length"]),
            padding_id=int(arrays["padding_id"]),
            oov_id=int(arrays["oov_id"]),
            game_table=arrays["game_table"].astype(np.float32, copy=False),
            seen_games=arrays["seen_games"].astype(bool, copy=False),
            w1=arrays["w1"],
            b1=arrays["b1"],
            w2=arrays["w2"],
            b2=arrays["b2"],
        )

    def embed(self, history: np.ndarray) -> np.ndarray:
        """history: int64 [history_length] game_idx, most recent first, padding-filled."""
        vocab = len(self.game_table)
        ids = np.where(history < vocab, history, self.oov_id)
        ids = np.where(self.seen_games[ids] | (ids == self.padding_id), ids, self.oov_id)
        present = ids != self.padding_id
        length = float(present.sum())
        pooled = self.game_table[ids[present]].sum(axis=0) / max(length, 1.0)
        x = np.concatenate([pooled, [length / self.history_length]]).astype(np.float32)
        hidden = np.maximum(x @ self.w1.T + self.b1, 0.0)
        out = hidden @ self.w2.T + self.b2
        return (out / max(float(np.linalg.norm(out)), EPS)).astype(np.float32)


class OnlineModel:
    def __init__(
        self, manifest: BundleManifest, catalog: dict[str, np.ndarray], user_tower: UserTower
    ) -> None:
        if user_tower.model_id != manifest.model_id:
            raise ValueError(f"user tower of {user_tower.model_id}, catalog of {manifest.model_id}")
        self.manifest = manifest
        self.user_tower = user_tower
        self.item_embeddings = catalog["item_embeddings"].astype(np.float32, copy=False)
        self.item_game_id = catalog["item_game_id"].astype(np.int64, copy=False)
        self._row_game_idx = catalog["item_game_idx"].astype(np.int64, copy=False)
        self._name_bytes = catalog["item_name_utf8"].tobytes()
        self._name_offsets = catalog["item_name_offsets"].astype(np.int64, copy=False)
        # game_id -> catalog row (sorted for searchsorted)
        order = np.argsort(self.item_game_id, kind="stable")
        self._sorted_ids = self.item_game_id[order]
        self._sorted_rows = order

    @classmethod
    def from_bytes(
        cls, manifest: BundleManifest, catalog_payload: bytes, user_tower: UserTower
    ) -> OnlineModel:
        if manifest.format_version != SUPPORTED_MANIFEST_FORMAT:
            raise ValueError(
                f"bundle format {manifest.format_version}, "
                f"this code reads {SUPPORTED_MANIFEST_FORMAT}"
            )
        if hashlib.sha256(catalog_payload).hexdigest() != manifest.catalog_sha256:
            raise ValueError("catalog sha256 does not match its manifest")
        catalog = _load_npz(catalog_payload, CATALOG_ARRAYS, "catalog")
        return cls(manifest, catalog, user_tower)

    @property
    def catalog_games(self) -> int:
        return len(self.item_game_id)

    def rows_of(self, game_ids: np.ndarray) -> np.ndarray:
        """Catalog row of each game id, -1 when not in the catalog."""
        pos = np.searchsorted(self._sorted_ids, game_ids)
        pos = np.clip(pos, 0, len(self._sorted_ids) - 1)
        found = self._sorted_ids[pos] == game_ids
        return np.where(found, self._sorted_rows[pos], -1)

    def name(self, row: int) -> str:
        start, end = self._name_offsets[row], self._name_offsets[row + 1]
        return self._name_bytes[start:end].decode()

    def recommend(self, liked_game_ids: list[int], k: int) -> OnlineResult:
        """Top k catalog games for a history of liked games (most recent first). Liked games are
        never recommended; unknown ones are ignored. Empty when no liked game is known."""
        tower = self.user_tower
        liked = np.asarray(list(dict.fromkeys(liked_game_ids)), dtype=np.int64)
        rows = self.rows_of(liked)
        known = rows >= 0
        ignored = liked[~known].tolist()
        if not known.any():
            return OnlineResult([], [], [], [], ignored)
        used_rows = rows[known][: tower.history_length]
        history = np.full(tower.history_length, tower.padding_id, dtype=np.int64)
        history[: len(used_rows)] = self._row_game_idx[used_rows]
        scores = self.item_embeddings @ tower.embed(history)
        scores[rows[known]] = -np.inf  # every liked game, not only the first history_length
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.zeros(0, dtype=np.int64)
        top = top[np.lexsort((self.item_game_id[top], -scores[top]))]  # ties: lowest appid
        return OnlineResult(
            game_ids=self.item_game_id[top].tolist(),
            names=[self.name(int(row)) for row in top],
            scores=[round(float(s), 4) for s in scores[top]],
            used_game_ids=self.item_game_id[used_rows].tolist(),
            ignored_game_ids=ignored,
        )


def _load_npz(payload: bytes, names: tuple[str, ...], what: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        missing = [name for name in names if name not in npz.files]
        if missing:
            raise ValueError(f"{what} is missing {missing}")
        return {name: npz[name] for name in names}


# The catalog is ~27 MB: allow a longer read than the DynamoDB calls.
S3_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=2, read_timeout=10
)


class BundleLoader:
    """Loads the model from S3 on first use, then re-reads the manifest at most every
    `refresh_seconds` and reloads when it changed (the user tower only when the model did).
    A failed refresh keeps the loaded model."""

    def __init__(self, bucket: str, prefix: str, *, region: str, refresh_seconds: int, s3=None):
        self.bucket = bucket
        self.manifest_key = f"{prefix.strip('/')}/manifest.json"
        self.refresh_seconds = refresh_seconds
        self.s3 = s3 or boto3.client("s3", region_name=region, config=S3_CONFIG)
        self._model: OnlineModel | None = None
        self._checked_at = float("-inf")

    def get(self) -> OnlineModel | None:
        """The current model, or None when nothing was ever published."""
        now = time.monotonic()
        if self._model is not None and now - self._checked_at < self.refresh_seconds:
            return self._model
        self._checked_at = now
        try:
            self._refresh()
        except self.s3.exceptions.NoSuchKey:
            log.warning("no online manifest at s3://%s/%s", self.bucket, self.manifest_key)
        except Exception:
            if self._model is None:
                raise
            log.exception("online refresh failed: keeping %s", self._model.manifest.model_id)
        return self._model

    def _get(self, key: str, version_id: str | None = None) -> bytes:
        request = {"Bucket": self.bucket, "Key": key}
        if version_id:
            request["VersionId"] = version_id
        return self.s3.get_object(**request)["Body"].read()

    def _refresh(self) -> None:
        manifest = BundleManifest.model_validate_json(self._get(self.manifest_key))
        current = self._model
        if current is not None and current.manifest == manifest:
            return
        started = time.perf_counter()
        user_tower = (
            current.user_tower
            if current is not None and current.manifest.user_tower_key == manifest.user_tower_key
            else UserTower.from_bytes(self._get(manifest.user_tower_key))
        )
        catalog = self._get(manifest.catalog_key, manifest.catalog_version_id)
        self._model = OnlineModel.from_bytes(manifest, catalog, user_tower)
        log.info(
            "loaded the online model %s (%d games) in %.2fs",
            manifest.model_id,
            self._model.catalog_games,
            time.perf_counter() - started,
        )
