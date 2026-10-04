"""Local parquet cache for slow full-data results (the game catalog, monthly counts, …).

`cached("catalog", game_catalog)` builds once and reads `<cache_dir>/catalog.parquet` after
that. The marts change on every pipeline run: pass `refresh=True` (or delete the directory)
to rebuild. The cache is local only (`notebooks/.cache/`, gitignored).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from steam_eda.sources import settings


def cache_path(key: str) -> Path:
    return Path(settings().cache_dir) / f"{key}.parquet"


def cached(key: str, build: Callable[[], pl.DataFrame], refresh: bool = False) -> pl.DataFrame:
    path = cache_path(key)
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df
