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
Format 3 (A3) appends, before the tag profile (present iff its arrays are), a self-attention
summary of the history: tokens = [attn_summary, game_table[ids] + attn_position[slot] for the
non-padding slots], then per block x += MHA(LN1(x)), x += FFN(LN2(x)) (ReLU), and the summary
token's LayerNorm-ed final state (training `HistoryAttention`; padding is masked out of every
block there, so dropping it here is exact).
Scores are cosine similarities with the (already normalized) item embeddings. The inference
test `test_online_parity.py` checks this against torch and batch retrieval.

Spec 9: when the manifest names a ranker (`ranker.npz`, written by inference with the catalog,
only for the serving pair's ranker of this model), the top `candidates` retrieved games are
re-ordered by it (`ranker_mode` "on") or only compared with retrieval order and logged
("shadow"). A ranker that fails to load or to rank keeps retrieval order.
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

from steam_serving.ranker import OnlineRanker

log = logging.getLogger(__name__)

SUPPORTED_MANIFEST_FORMAT = 2
# 1: plain user tower; 2: + the A4 tag profile; 3: + the A3 attention encoder, tag profile
# optional (training `export.py`)
SUPPORTED_USER_TOWER_FORMATS = (1, 2, 3)
TAG_PROFILE_FORMAT = 2
ATTENTION_FORMAT = 3
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
ATTENTION_ARRAYS = (
    "attn_heads",
    "attn_ln_eps",
    "attn_summary",
    "attn_position",
    "attn_norm_w",
    "attn_norm_b",
    "attn_ln1_w",
    "attn_ln1_b",
    "attn_qkv_w",
    "attn_qkv_b",
    "attn_out_w",
    "attn_out_b",
    "attn_ln2_w",
    "attn_ln2_b",
    "attn_ffn_w1",
    "attn_ffn_b1",
    "attn_ffn_w2",
    "attn_ffn_b2",
)


@dataclass(frozen=True)
class HistoryAttention:
    """Format 3: training's `HistoryAttention` for one history (arrays as exported)."""

    heads: int
    eps: float
    arrays: dict[str, np.ndarray]  # ATTENTION_ARRAYS, float32 (per block: leading layer axis)

    @property
    def layers(self) -> int:
        return len(self.arrays["attn_qkv_w"])

    def _norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * weight + bias

    def summarize(self, games: np.ndarray, slots: np.ndarray) -> np.ndarray:
        """games: float32 [n, d], the non-padding history games; slots: their positions."""
        a = self.arrays
        x = np.concatenate([a["attn_summary"][None], games + a["attn_position"][slots]])
        tokens, dim = x.shape
        head_dim = dim // self.heads
        for i in range(self.layers):
            h = self._norm(x, a["attn_ln1_w"][i], a["attn_ln1_b"][i])
            qkv = h @ a["attn_qkv_w"][i].T + a["attn_qkv_b"][i]
            q, k, v = (
                part.reshape(tokens, self.heads, head_dim).transpose(1, 0, 2)
                for part in np.split(qkv, 3, axis=-1)
            )
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = scores / scores.sum(axis=-1, keepdims=True)
            mixed = (weights @ v).transpose(1, 0, 2).reshape(tokens, dim)
            x = x + mixed @ a["attn_out_w"][i].T + a["attn_out_b"][i]
            h = self._norm(x, a["attn_ln2_w"][i], a["attn_ln2_b"][i])
            hidden = np.maximum(h @ a["attn_ffn_w1"][i].T + a["attn_ffn_b1"][i], 0.0)
            x = x + hidden @ a["attn_ffn_w2"][i].T + a["attn_ffn_b2"][i]
        return self._norm(x[0], a["attn_norm_w"], a["attn_norm_b"]).astype(np.float32)


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
    # spec 9: the serving pair's ranker (None: retrieval order)
    ranker_id: str | None = None
    ranker_mode: str = "off"  # inference's RANKER_MODE: "on" re-orders, "shadow" only compares
    ranker_key: str | None = None
    ranker_version_id: str | None = None
    ranker_sha256: str | None = None


@dataclass(frozen=True)
class OnlineResult:
    game_ids: list[int]  # best first
    names: list[str]
    scores: list[float]  # cosine similarity
    used_game_ids: list[int]  # liked games that made up the history (at most history_length)
    ignored_game_ids: list[int]  # liked games the model does not know (not in the catalog)
    ranker_id: str | None = None  # the ranker that ordered the list (None: retrieval order)


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
    # format 3 only (A3 attention encoder), else None
    attention: HistoryAttention | None = None

    @classmethod
    def from_bytes(cls, payload: bytes) -> UserTower:
        arrays = _load_npz(payload, ("format_version",), "user tower")
        version = int(arrays["format_version"])
        if version not in SUPPORTED_USER_TOWER_FORMATS:
            raise ValueError(
                f"user tower format {version}, this code reads {SUPPORTED_USER_TOWER_FORMATS}"
            )
        with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
            files = set(npz.files)
        # format 2 always has the tag profile; format 3 when its arrays are there
        has_profile = version == TAG_PROFILE_FORMAT or (
            version >= ATTENTION_FORMAT and "tag_table" in files
        )
        names = USER_TOWER_ARRAYS + (TAG_PROFILE_ARRAYS if has_profile else ())
        if version >= ATTENTION_FORMAT:
            names += ATTENTION_ARRAYS
        arrays = _load_npz(payload, names, "user tower")
        profile: dict = {}
        if version >= ATTENTION_FORMAT:
            profile["attention"] = HistoryAttention(
                heads=int(arrays["attn_heads"]),
                eps=float(arrays["attn_ln_eps"]),
                arrays={
                    name: arrays[name].astype(np.float32, copy=False)
                    for name in ATTENTION_ARRAYS[2:]
                },
            )
        if has_profile:
            profile |= {
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
        if self.attention is not None:
            slots = np.flatnonzero(present)
            parts.append(self.attention.summarize(self.game_table[ids[slots]], slots))
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
        self,
        manifest: BundleManifest,
        catalog: dict[str, np.ndarray],
        user_tower: UserTower,
        ranker: OnlineRanker | None = None,
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
        self.ranker = ranker
        if ranker is not None:
            check_ranker(ranker, manifest, self._row_game_idx)

    @classmethod
    def from_bytes(
        cls,
        manifest: BundleManifest,
        catalog_payload: bytes,
        user_tower: UserTower,
        ranker: OnlineRanker | None = None,
    ) -> OnlineModel:
        if manifest.format_version != SUPPORTED_MANIFEST_FORMAT:
            raise ValueError(
                f"bundle format {manifest.format_version}, "
                f"this code reads {SUPPORTED_MANIFEST_FORMAT}"
            )
        if hashlib.sha256(catalog_payload).hexdigest() != manifest.catalog_sha256:
            raise ValueError("catalog sha256 does not match its manifest")
        catalog = _load_npz(catalog_payload, CATALOG_ARRAYS, "catalog")
        return cls(manifest, catalog, user_tower, ranker)

    @property
    def ranker_mode(self) -> str:
        return self.manifest.ranker_mode if self.ranker is not None else "off"

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

    def recommend(
        self, liked_game_ids: list[int], k: int, now_seconds: float | None = None
    ) -> OnlineResult:
        """Top k catalog games for a history of liked games (most recent first). Liked games are
        never recommended; unknown ones are ignored. Empty when no liked game is known. With a
        ranker, its top `candidates` retrieved games are re-ordered (as of `now_seconds`,
        default now)."""
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
        mode = self.ranker_mode
        top = self._top(scores, max(k, self.ranker.candidates) if self.ranker else k)
        ranker_id = None
        if self.ranker is not None and mode in ("on", "shadow") and len(top):
            ranked = self._rank(rows[known], top, scores[top], now_seconds)
            if ranked is not None and mode == "on":
                top, ranker_id = ranked, self.ranker.ranker_id
            elif ranked is not None:
                overlap = len(set(top[:k].tolist()) & set(ranked[:k].tolist())) / len(top[:k])
                log.info("ranker shadow: overlap@%d %.2f", k, overlap)
        top = top[:k]
        return OnlineResult(
            game_ids=self.item_game_id[top].tolist(),
            names=[self.name(int(row)) for row in top],
            scores=[round(float(s), ROUND_DECIMALS) for s in scores[top]],
            used_game_ids=self.item_game_id[used_rows].tolist(),
            ignored_game_ids=ignored,
            ranker_id=ranker_id,
        )

    def _rank(
        self,
        history: np.ndarray,
        top: np.ndarray,
        scores: np.ndarray,
        now_seconds: float | None,
    ) -> np.ndarray | None:
        """The retrieved rows `top` in the ranker's order; None (retrieval order) on failure."""
        assert self.ranker is not None
        try:
            now = time.time() if now_seconds is None else now_seconds
            return top[self.ranker.order(history, top, scores, int(now))]
        except Exception:
            log.exception("ranker %s failed: retrieval order", self.ranker.ranker_id)
            return None

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


def check_ranker(ranker: OnlineRanker, manifest: BundleManifest, row_game_idx: np.ndarray) -> None:
    """A ranker serves only with the model and the catalog rows it was exported for."""
    if ranker.ranker_id != manifest.ranker_id:
        raise ValueError(
            f"ranker.npz holds {ranker.ranker_id}, the manifest names {manifest.ranker_id}"
        )
    if ranker.model_id != manifest.model_id:
        raise ValueError(
            f"ranker {ranker.ranker_id} serves with {ranker.model_id}, "
            f"the catalog is {manifest.model_id}'s"
        )
    if not np.array_equal(ranker.game_idx, row_game_idx):
        raise ValueError("ranker.npz rows differ from the catalog's")


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
        model = OnlineModel.from_bytes(manifest, catalog, user_tower)
        model.ranker = self._ranker(manifest, model)
        self._model = model
        log.info(
            "loaded the online model %s (%d games, ranker %s, %s) in %.2fs",
            manifest.model_id,
            model.catalog_games,
            manifest.ranker_id,
            model.ranker_mode,
            time.perf_counter() - started,
        )

    def _ranker(self, manifest: BundleManifest, model: OnlineModel) -> OnlineRanker | None:
        """The manifest's ranker, checked against the model; None (retrieval order) when there
        is none or it can't be used: a ranker problem never takes the endpoint down."""
        if manifest.ranker_key is None or manifest.ranker_mode == "off":
            return None
        try:
            payload = self._get(manifest.ranker_key, manifest.ranker_version_id)
            if hashlib.sha256(payload).hexdigest() != manifest.ranker_sha256:
                raise ValueError("ranker sha256 does not match its manifest")
            ranker = OnlineRanker.from_bytes(payload)
            check_ranker(ranker, manifest, model._row_game_idx)
            return ranker
        except Exception:
            log.exception("ranker %s not loaded: retrieval order", manifest.ranker_id)
            return None
