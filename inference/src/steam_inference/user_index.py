"""Demo user index (spec 13): the mart `user_index` (user_idx 1 = most active user) as a flat
file serving can read one user from with a byte-range request.

index.bin holds the pseudonymous user ids as little-endian int64 in user_idx order: user i sits
at byte offset (i - 1) * 8 (~8 MB per million users). Published with the online catalog by
`online.BundleStore.publish_user_index`, rewritten every run (user_idx is not stable across
runs). The mart's order (most reviews of kept games, ties by lowest user id) is the order of
`features.Activity.most_active` and `pipeline.select_rerank_users`, so the top of the index are
the scored and reranked users.
"""

from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
from pyiceberg.exceptions import NoSuchTableError
from steam_training.data import TableSource

log = logging.getLogger(__name__)

TABLE = "user_index"


def load_user_index(source: TableSource) -> np.ndarray | None:
    """int64 user ids, row i = user_idx i + 1; None when the mart is missing (ETL before spec
    13) or empty. Raises when user_idx is not exactly 1..count (dbt tests that too)."""
    try:
        snapshot_id = source.snapshot_id(TABLE)
    except NoSuchTableError:
        log.warning("mart %s not found: demo user index not published", TABLE)
        return None
    idx_parts, id_parts = [], []
    for batch in source.batches(TABLE, ["user_idx", "user_id"], snapshot_id):
        idx_parts.append(_int64(batch["user_idx"]))
        id_parts.append(_int64(batch["user_id"]))
    if not idx_parts or not sum(len(p) for p in idx_parts):
        log.warning("mart %s is empty: demo user index not published", TABLE)
        return None
    user_idx, user_id = np.concatenate(idx_parts), np.concatenate(id_parts)
    n = len(user_idx)
    if user_idx.min() != 1 or user_idx.max() != n or (np.bincount(user_idx - 1) != 1).any():
        raise ValueError(f"mart {TABLE}: user_idx is not exactly 1..{n} (snapshot {snapshot_id})")
    ordered = np.empty(n, dtype=np.int64)
    ordered[user_idx - 1] = user_id
    log.info("demo user index: %d users (snapshot %s)", n, snapshot_id)
    return ordered


def encode_user_index(user_ids: np.ndarray) -> bytes:
    return np.ascontiguousarray(user_ids, dtype="<i8").tobytes()


def _int64(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)
