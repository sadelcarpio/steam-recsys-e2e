"""A7 (spec 17): the language of every review, read straight from the raw reviews (no mart has
it yet), for the user tower's "language of the user's previous review" input.

Only `rec_id` (= the marts' `review_id`) and `language` are read from each raw parquet file.
Raw files are immutable once written, so with a cache directory each file's pairs are kept
locally (`<cache>/review-languages/<key>.npz`) and later runs only read the new files. If A7
is accepted, `language` belongs in `interactions` (an ETL column), and this module goes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from steam_training.contracts import OOV_ID, STEAM_LANGUAGES

log = logging.getLogger(__name__)

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
        """From (`rec_id`, `language`) batches; a review scraped twice keeps one copy."""
        ids, codes = [], []
        for batch in batches:
            ids.append(pc.fill_null(batch["rec_id"], -1).to_numpy(zero_copy_only=False))
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


def load_review_languages(uri: str, region: str, cache_dir: str | None = None) -> ReviewLanguages:
    """Every review's language under `uri` (s3://bucket/prefix/ of the raw reviews parquet)."""
    import boto3
    import pyarrow.parquet as pq
    from pyarrow import fs

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3", region_name=region)
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    filesystem = fs.S3FileSystem(region=region)
    root = Path(cache_dir, "review-languages", bucket) if cache_dir else None

    def read(key: str) -> tuple[np.ndarray, np.ndarray]:
        cached = root / f"{key}.npz" if root is not None else None
        if cached is not None and cached.exists():
            with np.load(cached) as npz:
                return npz["review_id"], npz["code"]
        table = pq.read_table(
            f"{bucket}/{key}", columns=["rec_id", "language"], filesystem=filesystem
        )
        ids = pc.fill_null(table["rec_id"], -1).to_numpy(zero_copy_only=False).astype(np.int64)
        codes = language_codes(table["language"])
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            part = cached.with_name(cached.name + ".part.npz")
            np.savez(part, review_id=ids, code=codes)
            part.replace(cached)
        return ids, codes

    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(read, keys))
    languages = ReviewLanguages.from_arrays(
        np.concatenate([i for i, _ in parts]) if parts else np.zeros(0, np.int64),
        np.concatenate([c for _, c in parts]) if parts else np.zeros(0, np.int8),
    )
    known = int((languages.code != OOV_ID).sum())
    log.info(
        "review languages: %d reviews from %d raw files (%d with a known language)",
        len(languages.review_id),
        len(keys),
        known,
    )
    return languages
