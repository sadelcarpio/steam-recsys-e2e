"""The ranker for the serving Lambda (spec 9, phase 5): `ranker.npz`, which numpy alone can run.

One file holds what `POST /recommendations` needs to re-order its retrieval candidates:

- the trees of a LightGBM ranker, flattened (`TREE_ARRAYS`; split features are indexes into
  `FEATURES`, whatever subset the ranker was trained on). The evaluator follows LightGBM's
  numerical decision: NaN becomes 0 unless the node's missing type is NaN; a missing value
  (zero for missing type Zero, NaN for NaN) goes to the default child; else `x <= threshold`
  goes left. The prediction is the sum of the leaf values (lambdarank: no transform).
- the game side of the features for the rows of the published catalog (`GAME_ARRAYS`, same row
  order as `catalog.npz`): the popularity features as of the export (the weeks before its
  week), static attributes, and tags / genres / categories as CSR lists.

Serving's port (`serving/src/steam_serving/ranker.py`) computes the user side from the request's
liked games. Parity with `compute_features` and LightGBM is tested in inference's
`tests/test_online_ranker.py`. Bump `RANKER_NUMPY_FORMAT` when the layout changes, and
`FEATURE_SCHEMA_VERSION` when a feature does (serving checks both).
"""

from __future__ import annotations

import io

import numpy as np

from steam_training.ranker.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    POPULARITY_FEATURES,
    GameTable,
    PopularityTable,
    popularity_features,
    week_of,
)
from steam_training.ranker.model import FitResult

RANKER_NUMPY_FORMAT = 1
MISSING_TYPES = {"None": 0, "Zero": 1, "NaN": 2}
TREE_ARRAYS = (
    "tree_roots",  # int32 [trees] node index of each tree's root
    "node_feature",  # int32 [nodes] index into FEATURES; -1 = leaf
    "node_threshold",  # float64 [nodes]
    "node_left",  # int32 [nodes]
    "node_right",  # int32 [nodes]
    "node_default_left",  # bool [nodes]
    "node_missing",  # int8 [nodes] MISSING_TYPES
    "node_value",  # float64 [nodes] leaf value (0 for internal nodes)
    "max_depth",  # int: edges on the longest root-leaf path
)
GAME_ARRAYS = (
    "game_idx",  # int64 [rows] the catalog row's game_idx (checked against catalog.npz)
    "game_popularity",  # float32 [rows, len(POPULARITY_FEATURES)]
    "game_is_free",  # float32 [rows]
    "game_log_price",  # float32 [rows], NaN = unknown
    "game_release_day",  # float32 [rows], days since the epoch, NaN = unknown
    "game_developer",  # int64 [rows], -1 = none
    "game_publisher",  # int64 [rows], -1 = none
    "game_tag_values",  # int32, tag ids (CSR)
    "game_tag_weights",  # float32, L2-normalized per game
    "game_tag_offsets",  # int64 [rows + 1]
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
    "features_utf8",  # ",".join(FEATURES), checked by serving
    "popularity_features_utf8",  # ",".join(POPULARITY_FEATURES)
    "ranker_id",
    "model_id",  # the retriever it serves with (the serving pair)
    "history_length",  # liked games read by the user side (the ranker's)
    "candidates",  # retrieval candidates it re-orders (inference's RETRIEVAL_CANDIDATES)
    "as_of_seconds",  # the export time: popularity covers the weeks before its week
)


def tree_arrays(fit: FitResult) -> dict[str, np.ndarray]:
    """The trees LightGBM predicts with (`fit.best_iteration`), flattened (`TREE_ARRAYS`)."""
    dump = fit.booster.dump_model(num_iteration=fit.best_iteration)
    to_full = [FEATURES.index(name) for name in fit.features]
    nodes: dict[str, list] = {
        k: [] for k in ("feature", "threshold", "left", "right", "default_left", "missing", "value")
    }
    roots: list[int] = []
    max_depth = 0

    def add(node: dict, depth: int) -> int:
        nonlocal max_depth
        index = len(nodes["feature"])
        for values in nodes.values():
            values.append(0)
        if "split_index" not in node:  # leaf
            nodes["feature"][index] = -1
            nodes["value"][index] = float(node["leaf_value"])
            max_depth = max(max_depth, depth)
            return index
        if node["decision_type"] != "<=":
            raise ValueError(f"unsupported split {node['decision_type']!r} (categorical?)")
        nodes["feature"][index] = to_full[int(node["split_feature"])]
        nodes["threshold"][index] = float(node["threshold"])
        nodes["default_left"][index] = bool(node["default_left"])
        nodes["missing"][index] = MISSING_TYPES[node["missing_type"]]
        nodes["left"][index] = add(node["left_child"], depth + 1)
        nodes["right"][index] = add(node["right_child"], depth + 1)
        return index

    for tree in dump["tree_info"]:
        roots.append(add(tree["tree_structure"], 0))
    return {
        "tree_roots": np.asarray(roots, dtype=np.int32),
        "node_feature": np.asarray(nodes["feature"], dtype=np.int32),
        "node_threshold": np.asarray(nodes["threshold"], dtype=np.float64),
        "node_left": np.asarray(nodes["left"], dtype=np.int32),
        "node_right": np.asarray(nodes["right"], dtype=np.int32),
        "node_default_left": np.asarray(nodes["default_left"], dtype=bool),
        "node_missing": np.asarray(nodes["missing"], dtype=np.int8),
        "node_value": np.asarray(nodes["value"], dtype=np.float64),
        "max_depth": np.asarray(max_depth),
    }


def game_arrays(
    games: GameTable, popularity: PopularityTable, game_idx: np.ndarray, as_of_seconds: int
) -> dict[str, np.ndarray]:
    """The game side of the features for `game_idx` (the published catalog's rows, in order),
    popularity as of `as_of_seconds` (`GAME_ARRAYS`)."""
    game_idx = np.asarray(game_idx, dtype=np.int64)
    row = popularity.row(np.full(len(game_idx), week_of(as_of_seconds)))
    pop = popularity_features(popularity, game_idx, row)
    tag_values, tag_weights, tag_offsets = _csr(games.tags[game_idx])
    genre_values, _, genre_offsets = _csr(games.genres[game_idx])
    category_values, _, category_offsets = _csr(games.categories[game_idx])
    return {
        "game_idx": game_idx,
        "game_popularity": np.stack([pop[name] for name in POPULARITY_FEATURES], axis=1),
        "game_is_free": games.is_free[game_idx],
        "game_log_price": games.log_price[game_idx],
        "game_release_day": games.release_day[game_idx],
        "game_developer": games.developer[game_idx],
        "game_publisher": games.publisher[game_idx],
        "game_tag_values": tag_values,
        "game_tag_weights": tag_weights,
        "game_tag_offsets": tag_offsets,
        "game_genre_values": genre_values,
        "game_genre_offsets": genre_offsets,
        "game_category_values": category_values,
        "game_category_offsets": category_offsets,
        "num_tags": np.asarray(games.tags.shape[1]),
        "num_genres": np.asarray(games.genres.shape[1]),
        "num_categories": np.asarray(games.categories.shape[1]),
    }


def _csr(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(dense)
    offsets = np.zeros(len(dense) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(dense)), out=offsets[1:])
    return cols.astype(np.int32), dense[rows, cols].astype(np.float32), offsets


def ranker_arrays(
    fit: FitResult,
    *,
    ranker_id: str,
    model_id: str,
    history_length: int,
    candidates: int,
    games: GameTable,
    popularity: PopularityTable,
    game_idx: np.ndarray,
    as_of_seconds: int,
) -> dict[str, np.ndarray]:
    meta = {
        "format_version": np.asarray(RANKER_NUMPY_FORMAT),
        "feature_schema_version": np.asarray(FEATURE_SCHEMA_VERSION),
        "features_utf8": _utf8(",".join(FEATURES)),
        "popularity_features_utf8": _utf8(",".join(POPULARITY_FEATURES)),
        "ranker_id": np.asarray(ranker_id),
        "model_id": np.asarray(model_id),
        "history_length": np.asarray(history_length),
        "candidates": np.asarray(candidates),
        "as_of_seconds": np.asarray(as_of_seconds, dtype=np.int64),
    }
    arrays = meta | tree_arrays(fit) | game_arrays(games, popularity, game_idx, as_of_seconds)
    assert set(arrays) == set(META_ARRAYS + TREE_ARRAYS + GAME_ARRAYS)
    return arrays


def _utf8(text: str) -> np.ndarray:
    return np.frombuffer(text.encode(), dtype=np.uint8).copy()


def encode_ranker(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()
