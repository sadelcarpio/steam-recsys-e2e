"""A7 (spec 17): the language of every review, from the `review_languages` mart (one row per
`review_id`, the language it was first scraped with; reviews without one are absent), for the
user tower's "language of the user's previous review" input. Training, the ranker / pair
evaluation and inference read it; ids are `STEAM_LANGUAGES` + FIRST_ID (unknown: OOV_ID).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from steam_training.contracts import OOV_ID, STEAM_LANGUAGES

if TYPE_CHECKING:
    from steam_training.data import TableSource

log = logging.getLogger(__name__)

LANGUAGE_MART = "review_languages"
LANGUAGE_COLUMNS = ["review_id", "language"]
_CODE = {name: i + 2 for i, name in enumerate(STEAM_LANGUAGES)}  # 0 padding, 1 unknown


def language_codes(names: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Steam API language names -> ids (`STEAM_LANGUAGES` + 2; unknown or null -> OOV_ID)."""
    keys = pa.array(list(_CODE))
    index = pc.index_in(pc.utf8_lower(names), value_set=keys)
    position = pc.fill_null(index, -1).to_numpy(zero_copy_only=False)
    return np.where(position >= 0, position + 2, OOV_ID).astype(np.int8)


@dataclass(frozen=True)
class ReviewLanguages:
    """review_id -> language id, sorted by review_id."""

    review_id: np.ndarray  # int64, sorted, unique
    code: np.ndarray  # int8

    @classmethod
    def from_batches(cls, batches: Iterable[pa.RecordBatch | pa.Table]) -> ReviewLanguages:
        """From (`review_id`, `language`) batches; a review listed twice keeps one copy."""
        ids, codes = [], []
        for batch in batches:
            ids.append(pc.fill_null(batch["review_id"], -1).to_numpy(zero_copy_only=False))
            codes.append(language_codes(batch["language"]))
        return cls.from_arrays(
            np.concatenate(ids) if ids else np.zeros(0, np.int64),
            np.concatenate(codes) if codes else np.zeros(0, np.int8),
        )

    @classmethod
    def from_arrays(cls, review_id: np.ndarray, code: np.ndarray) -> ReviewLanguages:
        keep = review_id >= 0
        review_id, code = review_id[keep].astype(np.int64), code[keep]
        review_id, first = np.unique(review_id, return_index=True)
        return cls(review_id, code[first])

    def of(self, review_id: np.ndarray) -> np.ndarray:
        """Language id of each review (OOV_ID when the review or its language is unknown)."""
        if len(self.review_id) == 0:
            return np.full(len(review_id), OOV_ID, np.int8)
        at = np.clip(np.searchsorted(self.review_id, review_id), 0, len(self.review_id) - 1)
        return np.where(self.review_id[at] == review_id, self.code[at], OOV_ID).astype(np.int8)


def load_review_languages(
    source: TableSource,
    snapshot_id: int | None = None,
    review_ids: np.ndarray | None = None,
) -> ReviewLanguages:
    """The `review_languages` mart (at `snapshot_id`, default: current). `review_ids` (sorted,
    unique): keep only those reviews (inference needs one per user, not the ~100M)."""
    if snapshot_id is None:
        snapshot_id = source.snapshot_id(LANGUAGE_MART)
    ids, codes = [], []
    for batch in source.batches(LANGUAGE_MART, LANGUAGE_COLUMNS, snapshot_id):
        review_id = pc.fill_null(batch["review_id"], -1).to_numpy(zero_copy_only=False)
        review_id = review_id.astype(np.int64)
        code = language_codes(batch["language"])
        if review_ids is not None:
            at = np.clip(np.searchsorted(review_ids, review_id), 0, max(len(review_ids) - 1, 0))
            keep = review_ids[at] == review_id if len(review_ids) else np.zeros(0, bool)
            review_id, code = review_id[keep], code[keep]
        ids.append(review_id)
        codes.append(code)
    languages = ReviewLanguages.from_arrays(
        np.concatenate(ids) if ids else np.zeros(0, np.int64),
        np.concatenate(codes) if codes else np.zeros(0, np.int8),
    )
    log.info(
        "review languages: %d reviews (%d with a known language)",
        len(languages.review_id),
        int((languages.code != OOV_ID).sum()),
    )
    return languages
