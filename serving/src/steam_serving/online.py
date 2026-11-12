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
Format 2 (A4 tag profile) appends a profile to x: the history games' tags (raw game_idx, so
games without training interactions count too; beyond the vocabulary: none), each weighted by
its Steam tag weight * exp(tag_recency[position]), normalized to sum 1 over the history, as a
weighted sum of tag_table rows (tags beyond the table: oov_id).
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
# 1: plain user tower; 2: + the A4 tag profile (training `export.py`)
SUPPORTED_USER_TOWER_FORMATS = (1, 2)
TAG_PROFILE_FORMAT = 2
EPS = 1e-12  # torch.nn.functional.normalize
ROUND_DECIMALS = 4  # returned scores; ranking ties at this precision go to the lowest appid
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
TAG_PROFILE_ARRAYS = ("tag_values", "tag_offsets", "tag_weights", "tag_recency", "tag_table")


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
    # format 2 only (A4 tag profile), else None
    tag_values: np.ndarray | None = None  # int64, game_idx -> tags (CSR)
    tag_offsets: np.ndarray | None = None  # int64 [vocab + 1]
    tag_weights: np.ndarray | None = None  # float32
    tag_recency: np.ndarray | None = None  # float32 [history_length], weight = exp
    tag_table: np.ndarray | None = None  # float32 [tags, dim]

    @classmethod
    def from_bytes(cls, payload: bytes) -> UserTower:
        arrays = _load_npz(payload, ("format_version",), "user tower")
        version = int(arrays["format_version"])
        if version not in SUPPORTED_USER_TOWER_FORMATS:
            raise ValueError(
                f"user tower format {version}, this code reads {SUPPORTED_USER_TOWER_FORMATS}"
            )
        names = USER_TOWER_ARRAYS + (TAG_PROFILE_ARRAYS if version >= TAG_PROFILE_FORMAT else ())
        arrays = _load_npz(payload, names, "user tower")
        profile = {}
        if version >= TAG_PROFILE_FORMAT:
            profile = {
                "tag_values": arrays["tag_values"].astype(np.int64, copy=False),
                "tag_offsets": arrays["tag_offsets"].astype(np.int64, copy=False),
                "tag_weights": arrays["tag_weights"].astype(np.float32, copy=False),
                "tag_recency": arrays["tag_recency"].astype(np.float32, copy=False),
                "tag_table": arrays["tag_table"].astype(np.float32, copy=False),
            }
        return cls(
            **profile,
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
        parts = [pooled, [length / self.history_length]]
        if self.tag_table is not None:
            parts.append(self._tag_profile(history))
        x = np.concatenate(parts).astype(np.float32)
        hidden = np.maximum(x @ self.w1.T + self.b1, 0.0)
        out = hidden @ self.w2.T + self.b2
        return (out / max(float(np.linalg.norm(out)), EPS)).astype(np.float32)

    def _tag_profile(self, history: np.ndarray) -> np.ndarray:
        """torch `UserTower._tag_profile` for one history."""
        assert self.tag_offsets is not None and self.tag_values is not None
        assert self.tag_weights is not None and self.tag_recency is not None
        assert self.tag_table is not None
        vocab = len(self.game_table)
        games = np.where(history < vocab, history, self.oov_id)  # raw ids: unseen games count
        recency = np.exp(self.tag_recency).astype(np.float32)
        values, weights = [], []
        for slot, game in enumerate(games.tolist()):
            lo, hi = int(self.tag_offsets[game]), int(self.tag_offsets[game + 1])
            values.append(self.tag_values[lo:hi])
            weights.append(self.tag_weights[lo:hi] * recency[slot])
        tags = np.concatenate(values) if values else np.zeros(0, np.int64)
        if len(tags) == 0:
            return np.zeros(self.tag_table.shape[1], dtype=np.float32)
        w = np.concatenate(weights).astype(np.float32)
        w = w / max(float(w.sum()), EPS)
        tags = np.where(tags < len(self.tag_table), tags, self.oov_id)
        return (self.tag_table[tags] * w[:, None]).sum(axis=0).astype(np.float32)


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
        top = self._top(scores, k)
        return OnlineResult(
            game_ids=self.item_game_id[top].tolist(),
            names=[self.name(int(row)) for row in top],
            scores=[round(float(s), ROUND_DECIMALS) for s in scores[top]],
            used_game_ids=self.item_game_id[used_rows].tolist(),
            ignored_game_ids=ignored,
        )

    def _top(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Rows of the k best scores, ranked by the score as returned (ROUND_DECIMALS) and then
        by lowest appid: identical games (e.g. OOV games with the same features) differ only by
        float noise that depends on the CPU / BLAS, so ranking raw floats is not reproducible.
        Rounds `scores` in place."""
        np.round(scores, ROUND_DECIMALS, out=scores)
        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        kth = scores[np.argpartition(-scores, k - 1)[k - 1]]
        candidates = np.flatnonzero(scores >= kth)  # k rows + any ties with the k-th
        order = np.lexsort((self.item_game_id[candidates], -scores[candidates]))
        return candidates[order[:k]]


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
