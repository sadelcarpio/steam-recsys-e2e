"""Subsamples applied while streaming, so only the kept rows reach memory.

- `Sample.by_key("user_id", 0.01)`: every row of ~1% of the users (whole histories, the same
  users in every table keyed by user id: `interactions`, `user_features`, raw `author_id`).
  Representative of users, not of rows (heavy reviewers are as likely as anyone).
- `Sample.rows(0.01)`: ~1% of the rows, uniform. Representative of rows, breaks histories.

Both are deterministic for a `seed`: the key sample hashes the key, the row sample hashes the
row position within the stream. Polars hashes are only stable for one polars version
(pinned by uv.lock), so re-run a notebook after an upgrade instead of mixing saved samples.

Non-representative shortcuts live on the loaders: `where` (Iceberg pushdown, e.g. a time
window, prunes whole files) and `limit` (the first rows in file order).
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

_BUCKETS = 1_000_000


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["key", "rows"]
    fraction: float = Field(gt=0, le=1)
    key: str | None = None
    seed: int = 0

    @classmethod
    def by_key(cls, key: str, fraction: float, seed: int = 0) -> Sample:
        return cls(kind="key", key=key, fraction=fraction, seed=seed)

    @classmethod
    def rows(cls, fraction: float, seed: int = 0) -> Sample:
        return cls(kind="rows", fraction=fraction, seed=seed)

    def keep(self, column: pl.Expr) -> pl.Expr:
        return (column.hash(self.seed) % _BUCKETS) < int(self.fraction * _BUCKETS)

    def apply(self, df: pl.DataFrame, offset: int = 0) -> pl.DataFrame:
        """Rows of `df` in the sample; `offset` is the position of `df`'s first row in the
        stream (row samples only)."""
        if self.fraction >= 1:
            return df
        if self.kind == "key":
            if self.key not in df.columns:
                raise ValueError(f"sample key {self.key!r} is not a loaded column")
            return df.filter(self.keep(pl.col(self.key)))
        position = pl.int_range(offset, offset + df.height, dtype=pl.UInt64)
        return df.filter(self.keep(position))
