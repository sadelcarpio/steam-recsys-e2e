"""Loaders: the Iceberg marts (Glue `steam_marts`) and the raw scraped parquet.

```python
from steam_eda import Sample, load_mart, load_raw

# 1% of the users, whole histories, only positives, 2025 onwards
df = load_mart(
    "interactions",
    columns=["user_id", "game_idx", "timestamp", "is_positive"],
    where="timestamp >= '2025-01-01T00:00:00'",  # Iceberg pushdown: skips whole files
    filter=pl.col("is_positive"),                 # polars, per batch
    sample=Sample.by_key("user_id", 0.01),
)
```

Cost model: `where` prunes files by their column statistics (cheap), `filter` / `sample` read
the selected columns of every remaining file and keep only the matching rows (memory ~ kept
rows), `limit` stops reading once enough rows are kept. Select only the columns you need:
the list columns (developers, genres, …) and raw review texts dominate the bytes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from typing import Any

import boto3
import polars as pl
import pyarrow as pa

from steam_eda.config import EdaSettings
from steam_eda.sampling import Sample

RAW_TABLES = ("reviews", "games", "game_tags")
# The raw review texts are ~90% of the raw reviews' bytes: never loaded unless asked for.
RAW_HEAVY_COLUMNS = {
    "reviews": {"review"},
    "games": {"detailed_description", "about_the_game", "minimum_pc_requirements",
              "recommended_pc_requirements"},
}  # fmt: skip


@cache
def settings() -> EdaSettings:
    return EdaSettings()


@cache
def _catalog(region: str):
    from pyiceberg.catalog import load_catalog

    return load_catalog("glue", type="glue", **{"glue.region": region})


def mart_table(name: str):
    """The pyiceberg table of a mart (schema, snapshots, `inspect` metadata tables)."""
    s = settings()
    return _catalog(s.aws_region).load_table((s.marts_database, name))


def mart_schema(name: str) -> pl.Schema:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return pl.from_arrow(schema_to_pyarrow(mart_table(name).schema()).empty_table()).schema


def collect(
    batches: Iterable[pa.RecordBatch],
    filter: pl.Expr | None = None,
    sample: Sample | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Streams `batches` into one DataFrame, keeping only the rows that pass `sample` then
    `filter`, and stops after `limit` kept rows."""
    kept: list[pl.DataFrame] = []
    seen = rows = 0
    schema = None
    for batch in batches:
        df = pl.from_arrow(batch)
        assert isinstance(df, pl.DataFrame)
        schema = df.schema
        offset, seen = seen, seen + df.height
        if sample is not None:
            df = sample.apply(df, offset)
        if filter is not None:
            df = df.filter(filter)
        if limit is not None:
            df = df.head(limit - rows)
        if df.height:
            kept.append(df)
            rows += df.height
        if limit is not None and rows >= limit:
            break
    if not kept:
        return pl.DataFrame(schema=schema)
    return pl.concat(kept, rechunk=True)


def load_mart(
    name: str,
    columns: list[str] | None = None,
    where: str | Any | None = None,
    filter: pl.Expr | None = None,
    sample: Sample | None = None,
    limit: int | None = None,
    snapshot_id: int | None = None,
) -> pl.DataFrame:
    """A mart as a polars DataFrame. `where` is an Iceberg row filter (string such as
    "timestamp >= '2025-01-01T00:00:00' and is_positive = true", or a pyiceberg expression).
    `snapshot_id` pins an older version (time travel); default: the current snapshot."""
    table = mart_table(name)
    return collect(_iceberg_batches(table, columns, where, snapshot_id), filter, sample, limit)


def iter_mart(
    name: str,
    columns: list[str] | None = None,
    where: str | Any | None = None,
    snapshot_id: int | None = None,
) -> Iterator[pl.DataFrame]:
    """A mart as a stream of DataFrames (`EdaSettings.batch_rows` rows at most each), for
    aggregations over the full data that would not fit in memory as one frame."""
    for batch in _iceberg_batches(mart_table(name), columns, where, snapshot_id):
        df = pl.from_arrow(batch)
        assert isinstance(df, pl.DataFrame)
        yield df


def _iceberg_batches(table, columns, where, snapshot_id) -> Iterator[pa.RecordBatch]:
    """Streams one data file after another (the training's `IcebergSource` approach):
    pyiceberg's public readers open every file of the scan at once and buffer them fully,
    tens of GB for `interactions` / `game_features`."""
    from pyiceberg.expressions import AlwaysTrue
    from pyiceberg.io.pyarrow import ArrowScan, _read_all_delete_files, schema_to_pyarrow

    scan = table.scan(
        row_filter=where if where is not None else AlwaysTrue(),
        selected_fields=tuple(columns) if columns else ("*",),
        snapshot_id=snapshot_id,
    )
    tasks = list(scan.plan_files())
    projected = scan.projection()
    target = schema_to_pyarrow(projected)
    arrow = ArrowScan(scan.table_metadata, scan.io, projected, scan.row_filter, scan.case_sensitive)
    batches = arrow._record_batches_from_scan_tasks_and_deletes(
        tasks, _read_all_delete_files(scan.io, tasks)
    )
    for batch in pa.RecordBatchReader.from_batches(target, batches).cast(target):
        for start in range(0, batch.num_rows, settings().batch_rows):
            yield batch.slice(start, settings().batch_rows)


def raw_files(table: str, last: int | None = None) -> list[str]:
    """Raw parquet files of `table`, oldest first (keys start with the scrape date);
    `last=N` keeps the N newest (a quick, recent-biased subset)."""
    if table not in RAW_TABLES:
        raise ValueError(f"unknown raw table {table!r}, expected one of {RAW_TABLES}")
    root = settings().raw_location
    if root.startswith("s3://"):
        bucket, _, prefix = root.removeprefix("s3://").partition("/")
        prefix = f"{prefix}/{table}/".lstrip("/")
        pages = (
            boto3.client("s3")
            .get_paginator("list_objects_v2")
            .paginate(Bucket=bucket, Prefix=prefix)
        )
        keys = [o["Key"] for p in pages for o in p.get("Contents", [])]
        files = sorted(f"s3://{bucket}/{k}" for k in keys if k.endswith(".parquet"))
    else:
        from pathlib import Path

        files = sorted(str(p) for p in Path(root, table).glob("*.parquet"))
    return files[-last:] if last else files


def scan_raw(table: str, last_files: int | None = None) -> pl.LazyFrame:
    """Lazy scan of the raw parquet (predicate / projection pushdown). The raw data is
    append-only: a review or game scraped twice has two rows (dedupe on `rec_id` / `appid`,
    keeping the latest `scrape_date`)."""
    files = raw_files(table, last_files)
    if not files:
        raise FileNotFoundError(f"no parquet files for raw table {table!r}")
    options = {"aws_region": settings().aws_region} if files[0].startswith("s3://") else None
    return pl.scan_parquet(files, storage_options=options, missing_columns="insert",
                           extra_columns="ignore")  # fmt: skip


def load_raw(
    table: str,
    columns: list[str] | None = None,
    filter: pl.Expr | None = None,
    sample: Sample | None = None,
    limit: int | None = None,
    last_files: int | None = None,
) -> pl.DataFrame:
    """A raw table as a DataFrame. Without `columns`, every column but the heavy texts
    (`RAW_HEAVY_COLUMNS`). Key samples only (`Sample.by_key("author_id", …)`): row samples need
    a stream position, use `load_mart`-style batches or `.sample()` after loading."""
    lf = scan_raw(table, last_files)
    if columns is None:
        heavy = RAW_HEAVY_COLUMNS.get(table, set())
        columns = [c for c in lf.collect_schema().names() if c not in heavy]
    lf = lf.select(columns)
    if sample is not None:
        if sample.kind != "key":
            raise ValueError("load_raw supports key samples only")
        lf = lf.filter(sample.keep(pl.col(sample.key)))
    if filter is not None:
        lf = lf.filter(filter)
    if limit is not None:
        lf = lf.head(limit)
    return lf.collect(engine="streaming")
