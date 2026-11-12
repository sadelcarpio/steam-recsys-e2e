"""Loaders over the public dataset files (spec 16): reviews, games, game_tags.

```python
from steam_eda import load_games, load_reviews

games = load_games()                      # one row per game + tags + parsed release date
reviews = load_reviews(fraction=0.05)     # whole histories of ~5% of the users, no texts
```

Reviews are 121M rows, ~90% of the bytes are the texts: `load_reviews` never reads `review`
unless it's asked for in `columns`. The user sample hashes `user_id`, so it is the same users
in every call (for a fixed polars version and seed).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from pathlib import Path

import polars as pl

from steam_eda.config import EdaSettings

TABLES = ("reviews", "games", "game_tags")
TEXT_COLUMNS = {"review"}
_BUCKETS = 1_000_000


@cache
def settings() -> EdaSettings:
    return EdaSettings()


def _matches(table: str, path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    parent = path.rsplit("/", 2)[-2] if path.count("/") >= 1 else ""
    return bool(re.fullmatch(rf"{table}(-\d+)?\.parquet", name)) or (
        parent == table and name.endswith(".parquet")
    )


def files(table: str, data_dir: str | None = None) -> list[str]:
    """Parquet files of `table`: `<table>.parquet`, `<table>-NNN.parquet` or `<table>/*.parquet`
    (the Athena UNLOAD parts under the S3 export prefix), sorted."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}, expected one of {TABLES}")
    root = (data_dir or settings().data_dir).rstrip("/")
    if root.startswith("s3://"):
        import boto3

        bucket, _, prefix = root.removeprefix("s3://").partition("/")
        prefix = f"{prefix}/" if prefix else ""
        pages = (
            boto3.client("s3", region_name=settings().aws_region)
            .get_paginator("list_objects_v2")
            .paginate(Bucket=bucket, Prefix=prefix)
        )
        paths = [f"s3://{bucket}/{o['Key']}" for p in pages for o in p.get("Contents", [])]
    else:
        paths = [str(p) for p in Path(root).rglob("*.parquet")]
    found = sorted(p for p in paths if _matches(table, p))
    if not found:
        raise FileNotFoundError(f"no parquet files for {table!r} under {root}")
    return found


def scan(table: str, data_dir: str | None = None) -> pl.LazyFrame:
    paths = files(table, data_dir)
    options = {"aws_region": settings().aws_region} if paths[0].startswith("s3://") else None
    return pl.scan_parquet(paths, storage_options=options)


def user_sample(fraction: float, seed: int = 0, column: str = "user_id") -> pl.Expr:
    """True for ~`fraction` of the users, by a hash of the id (whole histories kept)."""
    return (pl.col(column).hash(seed) % _BUCKETS) < int(fraction * _BUCKETS)


def load_reviews(
    fraction: float | None = None,
    columns: list[str] | None = None,
    filter: pl.Expr | None = None,
    seed: int | None = None,
    data_dir: str | None = None,
) -> pl.DataFrame:
    """Reviews of a hash sample of users (`fraction`, default `EDA_SAMPLE_USERS`; 1 = all).
    Without `columns`: every column but the text."""
    s = settings()
    lf = scan("reviews", data_dir)
    if columns is None:
        columns = [c for c in lf.collect_schema().names() if c not in TEXT_COLUMNS]
    fraction = s.sample_users if fraction is None else fraction
    if fraction < 1:
        lf = lf.filter(user_sample(fraction, s.seed if seed is None else seed))
    if filter is not None:
        lf = lf.filter(filter)
    return lf.select(columns).collect(engine="streaming")


def parse_release_date(text: pl.Expr) -> pl.Expr:
    """Steam's English release date ("Oct 9, 2020", "9 Oct, 2020") as a Date; month-only,
    year-only, "Q3 2026", "Coming soon", "To be announced" → null."""
    t = text.str.strip_chars()
    return pl.coalesce(
        t.str.to_date("%b %d, %Y", strict=False),
        t.str.to_date("%d %b, %Y", strict=False),
        t.str.to_date("%B %d, %Y", strict=False),
        t.str.to_date("%d %B, %Y", strict=False),
    )


def load_games(data_dir: str | None = None) -> pl.DataFrame:
    """One row per game: details + Steam user tags (by weight, descending) + `release_date`
    (parsed) and `release_year`. `game_price` is the price at the last scrape (USD)."""
    games = scan("games", data_dir).collect()
    tags = scan("game_tags", data_dir).select("game_id", "tag_names", "tag_weights").collect()
    return games.join(tags, on="game_id", how="left").with_columns(
        release_date=parse_release_date(pl.col("game_release_date")),
        release_year=pl.col("game_release_date")
        .str.extract(r"\b((?:19|20)\d{2})\b")
        .cast(pl.Int32, strict=False),
    )


def cached(key: str, build: Callable[[], pl.DataFrame], refresh: bool = False) -> pl.DataFrame:
    """`build()` once, then read `<cache_dir>/<key>.parquet`. Delete it (or `refresh=True`)
    after downloading a new dataset version."""
    path = Path(settings().cache_dir) / f"{key}.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df
