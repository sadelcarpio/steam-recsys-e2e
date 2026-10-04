# notebooks

EDA notebooks + the `steam_eda` helper package. Local only, no AWS writes, no CD. Human docs:
`README.md`.

## Layout

- `src/steam_eda/` (uv project, py3.12, polars; pandas only for seaborn plots)
  - `config.py`: `EdaSettings` (env only, no SSM): region, marts database, `raw_root`,
    `batch_rows`, `cache_dir`
  - `sources.py`: `load_mart` / `iter_mart` (Iceberg via Glue, streamed one file at a time with
    pyiceberg's private per-file iterator, same as training's `IcebergSource`), `collect`
    (filter / sample / limit over batches, pure, tested), `load_raw` / `scan_raw` / `raw_files`
    (polars lazy parquet, S3 or local)
  - `sampling.py`: `Sample.by_key` (hash of a key, consistent across tables) / `Sample.rows`
    (hash of the stream position, independent of batch sizes)
  - `catalog.py`: lookups + `decode`, `latest_game_features` (streamed reduce),
    `review_counts`, `game_catalog`, `release_year`
  - `features.py`: leakage-free `with_user_history` (strictly earlier reviews), `binned_rate`,
    `categorical_rate`, `single_feature_auc` / `feature_report`, `gini`, `lorenz`,
    `tag_profile_similarity`
  - `cache.py`: `cached(key, build)` parquet cache in `.cache/` (gitignored)
  - `plots.py`: seaborn setup, `rate_plot`, `loglog_hist`
- `*.ipynb`: committed **without outputs** (clear them before committing)
- `tests/`: unit tests on in-memory batches / local parquet (no AWS)

## Invariants

- Never load a full mart without `columns`: `interactions` and `game_features` hold 115M rows.
- History features must only use reviews strictly before the review (same rule as the marts).
- Dependencies through `uv add` only.
