"""The ranker of the serving pair (spec 9, phase 5), in numpy: re-orders the online retrieval
candidates of `POST /recommendations`.

`serving/online/ranker.npz` is written by inference with the catalog (training's
`steam_training.ranker.export`): the LightGBM trees, flattened, and the game side of the
features for the catalog rows (popularity as of the inference run). The user side comes from the
request's liked games, as training's `steam_training.ranker.features.compute_features` computes
it from a history (one group here). Parity with that function and with LightGBM is tested in
inference's `tests/test_online_ranker.py`; change the three together.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

SUPPORTED_RANKER_FORMAT = 1
# training's FEATURE_SCHEMA_VERSION / FEATURES / POPULARITY_FEATURES (checked on load)
FEATURE_SCHEMA_VERSION = 2
FEATURES = [
    "rank",
    "score_percentile",
    "score_gap",
    "log_positives",
    "log_positives_30d",
    "log_positives_90d",
    "positive_ratio",
    "is_free",
    "log_price",
    "age_days",
    "tag_cosine",
    "developer_share",
    "publisher_share",
    "genre_overlap",
    "category_overlap",
]
POPULARITY_FEATURES = ["log_positives", "log_positives_30d", "log_positives_90d", "positive_ratio"]
AFFINITY_FEATURES = FEATURES[FEATURES.index("tag_cosine") :]
SECONDS_PER_DAY = 86_400
MISSING_ZERO, MISSING_NAN = 1, 2
ZERO_THRESHOLD = 1e-35  # LightGBM kZeroThreshold
TREE_ARRAYS = (
    "tree_roots",
    "node_feature",
    "node_threshold",
    "node_left",
    "node_right",
    "node_default_left",
    "node_missing",
    "node_value",
    "max_depth",
)
GAME_ARRAYS = (
    "game_idx",
    "game_popularity",
    "game_is_free",
    "game_log_price",
    "game_release_day",
    "game_developer",
    "game_publisher",
    "game_tag_values",
    "game_tag_weights",
    "game_tag_offsets",
    "game_genre_values",
    "game_genre_offsets",
    "game_category_values",
    "game_category_offsets",
    "num_tags",
    "num_genres",
    "num_categories",
)
META_ARRAYS = (
    "format_version",
    "feature_schema_version",
    "features_utf8",
    "popularity_features_utf8",
    "ranker_id",
    "model_id",
    "history_length",
    "candidates",
    "as_of_seconds",
)


@dataclass(frozen=True)
class TreeEnsemble:
    """LightGBM's numerical decision, every tree at once for a matrix of rows."""

    roots: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    missing: np.ndarray
    value: np.ndarray
    max_depth: int

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray]) -> TreeEnsemble:
        return cls(
            roots=a["tree_roots"].astype(np.int64),
            feature=a["node_feature"].astype(np.int64),
            threshold=a["node_threshold"].astype(np.float64),
            left=a["node_left"].astype(np.int64),
            right=a["node_right"].astype(np.int64),
            default_left=a["node_default_left"].astype(bool),
            missing=a["node_missing"].astype(np.int8),
            value=a["node_value"].astype(np.float64),
            max_depth=int(a["max_depth"]),
        )

    @property
    def num_trees(self) -> int:
        return len(self.roots)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scores of rows x ([n, len(FEATURES)], NaN = missing)."""
        x = np.asarray(x, dtype=np.float64)
        node = np.broadcast_to(self.roots, (len(x), self.num_trees)).copy()
        rows = np.arange(len(x))[:, None]
        for _ in range(self.max_depth):
            feature = self.feature[node]
            inner = feature >= 0
            if not inner.any():
                break
            value = x[rows, np.maximum(feature, 0)]
            missing = self.missing[node]
            nan = np.isnan(value)
            value = np.where(nan & (missing != MISSING_NAN), 0.0, value)
            default = ((missing == MISSING_ZERO) & (np.abs(value) <= ZERO_THRESHOLD)) | (
                (missing == MISSING_NAN) & nan
            )
            with np.errstate(invalid="ignore"):
                left = np.where(default, self.default_left[node], value <= self.threshold[node])
            node = np.where(inner, np.where(left, self.left[node], self.right[node]), node)
        return self.value[node].sum(axis=1)


@dataclass(frozen=True)
class _Csr:
    values: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray | None = None

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(owner index into `rows`, position in values) of every entry of the given rows."""
        starts = self.offsets[rows]
        lengths = self.offsets[rows + 1] - starts
        owner = np.repeat(np.arange(len(rows)), lengths)
        first = np.repeat(np.cumsum(lengths) - lengths, lengths)
        return owner, np.arange(int(lengths.sum())) - first + np.repeat(starts, lengths)


@dataclass(frozen=True)
class OnlineRanker:
    ranker_id: str
    model_id: str  # the retriever it was evaluated with
    history_length: int
    candidates: int
    as_of_seconds: int
    trees: TreeEnsemble
    game_idx: np.ndarray  # per catalog row
    popularity: np.ndarray  # float32 [rows, len(POPULARITY_FEATURES)]
    is_free: np.ndarray
    log_price: np.ndarray
    release_day: np.ndarray
    developer: np.ndarray
    publisher: np.ndarray
    tags: _Csr
    genres: _Csr
    categories: _Csr
    num_tags: int
    num_genres: int
    num_categories: int

    @classmethod
    def from_bytes(cls, payload: bytes) -> OnlineRanker:
        with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
            missing = [n for n in META_ARRAYS + TREE_ARRAYS + GAME_ARRAYS if n not in npz.files]
            if missing:
                raise ValueError(f"ranker is missing {missing}")
            a = {name: npz[name] for name in META_ARRAYS + TREE_ARRAYS + GAME_ARRAYS}
        if int(a["format_version"]) != SUPPORTED_RANKER_FORMAT:
            raise ValueError(
                f"ranker format {int(a['format_version'])}, this code reads "
                f"{SUPPORTED_RANKER_FORMAT}"
            )
        if int(a["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"ranker feature schema {int(a['feature_schema_version'])}, this code computes "
                f"{FEATURE_SCHEMA_VERSION}"
            )
        if a["features_utf8"].tobytes().decode().split(",") != FEATURES or (
            a["popularity_features_utf8"].tobytes().decode().split(",") != POPULARITY_FEATURES
        ):
            raise ValueError("ranker features differ from this code's")
        return cls(
            ranker_id=str(a["ranker_id"]),
            model_id=str(a["model_id"]),
            history_length=int(a["history_length"]),
            candidates=int(a["candidates"]),
            as_of_seconds=int(a["as_of_seconds"]),
            trees=TreeEnsemble.from_arrays(a),
            game_idx=a["game_idx"].astype(np.int64),
            popularity=a["game_popularity"].astype(np.float32),
            is_free=a["game_is_free"].astype(np.float32),
            log_price=a["game_log_price"].astype(np.float32),
            release_day=a["game_release_day"].astype(np.float32),
            developer=a["game_developer"].astype(np.int64),
            publisher=a["game_publisher"].astype(np.int64),
            tags=_Csr(
                a["game_tag_values"].astype(np.int64),
                a["game_tag_offsets"].astype(np.int64),
                a["game_tag_weights"].astype(np.float32),
            ),
            genres=_Csr(
                a["game_genre_values"].astype(np.int64), a["game_genre_offsets"].astype(np.int64)
            ),
            categories=_Csr(
                a["game_category_values"].astype(np.int64),
                a["game_category_offsets"].astype(np.int64),
            ),
            num_tags=int(a["num_tags"]),
            num_genres=int(a["num_genres"]),
            num_categories=int(a["num_categories"]),
        )

    @property
    def rows(self) -> int:
        return len(self.game_idx)

    def features(
        self,
        history: np.ndarray,
        candidates: np.ndarray,
        scores: np.ndarray,
        now_seconds: int,
    ) -> np.ndarray:
        """float32 [candidates, len(FEATURES)] for one request. `history`: catalog rows of the
        liked games (most recent first, at most `history_length` are read), `candidates`:
        catalog rows best first, `scores`: their retrieval scores."""
        history = np.asarray(history, dtype=np.int64)[: self.history_length]
        candidates = np.asarray(candidates, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        k, n = len(candidates), len(history)
        out = np.full((k, len(FEATURES)), np.nan, dtype=np.float32)
        col = {name: i for i, name in enumerate(FEATURES)}

        rank = np.arange(k, dtype=np.float32)
        out[:, col["rank"]] = rank
        out[:, col["score_percentile"]] = 1 - rank / max(k - 1, 1)
        if k:
            top, lowest = scores[0], scores.min()
            out[:, col["score_gap"]] = (scores - top) / np.maximum(top - lowest, 1e-9)

        for i, name in enumerate(POPULARITY_FEATURES):
            out[:, col[name]] = self.popularity[candidates, i]
        out[:, col["is_free"]] = self.is_free[candidates]
        out[:, col["log_price"]] = self.log_price[candidates]
        day = np.float32(now_seconds / SECONDS_PER_DAY)
        out[:, col["age_days"]] = day - self.release_day[candidates]

        if n == 0:  # no history: the user side stays NaN
            return out
        profile = np.zeros(self.num_tags, dtype=np.float32)
        _, pos = self.tags.gather(history)
        assert self.tags.weights is not None
        np.add.at(profile, self.tags.values[pos], self.tags.weights[pos])
        profile /= max(float(np.linalg.norm(profile)), 1e-12)
        owner, pos = self.tags.gather(candidates)
        cosine = np.zeros(k, dtype=np.float32)
        np.add.at(cosine, owner, self.tags.weights[pos] * profile[self.tags.values[pos]])
        out[:, col["tag_cosine"]] = cosine
        for name, ids in (("developer_share", self.developer), ("publisher_share", self.publisher)):
            same = (ids[candidates][:, None] == ids[history][None, :]) & (
                ids[candidates][:, None] >= 0
            )
            out[:, col[name]] = same.sum(axis=1) / n
        for name, labels, size in (
            ("genre_overlap", self.genres, self.num_genres),
            ("category_overlap", self.categories, self.num_categories),
        ):
            share = np.zeros(size, dtype=np.float32)
            _, pos = labels.gather(history)
            np.add.at(share, labels.values[pos], 1.0)
            share /= n
            owner, pos = labels.gather(candidates)
            overlap = np.zeros(k, dtype=np.float32)
            np.add.at(overlap, owner, share[labels.values[pos]])
            count = np.bincount(owner, minlength=k).astype(np.float32)
            out[:, col[name]] = overlap / np.maximum(count, 1)
        return out

    def order(
        self, history: np.ndarray, candidates: np.ndarray, scores: np.ndarray, now_seconds: int
    ) -> np.ndarray:
        """Positions of `candidates`, best first by the ranker (ties keep retrieval order)."""
        score = self.trees.predict(self.features(history, candidates, scores, now_seconds))
        return np.argsort(-score, kind="stable")
