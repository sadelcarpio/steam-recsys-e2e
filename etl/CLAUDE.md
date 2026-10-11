# etl

Spec: `specs/2-data-transformation.md` (+ `specs/8-game-tags.md`, `specs/10-coming-soon-games.md`,
`specs/13-anonymize-user-ids.md`). Human docs: `README.md`.

## Layout

- `dbt/` (project `steam_recsys`, dbt-core 1.12 + dbt-athena, Athena engine v3 / Trino SQL)
  - `models/staging` (views), `models/intermediate` and `models/marts` (incremental Iceberg),
    `models/marts/lookups` (`lkp_*`, append-only, `full_refresh: false`), and `user_index`
    (spec 13), the only `table` materialization (rebuilt every run)
  - `macros/helpers.sql`: run stamp `batch_timestamp()`, reserved `padding_id()` (0) / `oov_id()` (1),
    `clean_string_array`, `encode_array`, `pad_ids`, `incremental_max`
  - `macros/asof.sql` (ASOF join CTEs), `macros/lookups.sql` (dense id vocabularies)
  - `tests/`: generic `unique_combination`, `expression_is_true` + singular tests (label
    leakage, `assert_user_index_is_dense`)
  - `profiles.yml`: env-var driven, one `athena` target
- `src/steam_etl/`
  - `config.py`: `EtlSettings` (env > SSM `/etl/*` when `USE_SSM=true`)
  - `runner.py`: `run_dbt()` maps settings to profile env / `--vars` and runs dbt in-process
  - `contracts.py`: Pydantic row contracts of every mart (`MART_CONTRACTS`)
  - `__main__.py`: ECS entry point (`dbt build`, exit 1 on any model/test failure)
- `tests/`: unit tests (no AWS), `tests/integration/` Athena end-to-end (`-m athena`)
- `Dockerfile`: image `etl`, dbt project at `/app/dbt`

## Invariants (keep them when changing models)

- Raw sources are Glue tables owned by Terraform (`infrastructure/etl.tf`, `local.raw_tables`),
  mirrored by `tests/integration/fixtures.py`. Update both when the scraper schema changes.
- Watermarks: `int_game_review_counts`, `user_features` and `interactions` consume
  `int_review_events` rows, and `game_features` consumes `int_game_review_counts` rows, with
  `_batch_at > max(_batch_at)` of the consuming model. `game_features` also re-merges every row
  of a game whose `int_games__deduplicated` row changed (same watermark), keeping the rows'
  `_batch_at`. Every row written must carry the
  `_batch_at` of the upstream rows it came from (or the run stamp).
- Lookups: never rebuild, never reuse ids. 0 = padding, 1 = OOV, first real id = 2. Padding and
  OOV must never be conflated (histories: 0 or >= 2; encoded arrays: >= 1).
- Features in `interactions` are strictly before the review (ASOF with events sorted before
  feature rows at equal timestamps).
- Raw review counts stay in `int_game_review_counts`; marts expose only the Laplace-smoothed ratio.
- `game_tags` (spec 8) is one row per catalog game with tags, merged on `game_id` only when a
  newer scrape (`scraped_at`) arrives, so a rerun writes nothing. `lkp_tags` covers the tags
  of `int_games__deduplicated` games only.
- `game_details` is display data, never a model feature. Its rows are the rows of
  `int_games__deduplicated` (merge key `game_name_key`, watermark `_batch_at`), with the text
  taken from the winning scrape. A re-scrape of a game that already wins its name does not
  update it (the details are treated as static; a full refresh takes the latest scrape),
  except when its `game_coming_soon` changed (spec 10: the released scrape replaces it).
- `game_name_key` is lower(name) of a game's **first** scrape (a stored winner keeps its key),
  so a rename (e.g. on release) never puts one appid under two keys and incremental equals
  full refresh.
- `user_id` is the scraper's pseudonymous id (spec 13), never a SteamID64: the raw column is
  `user_id`. `user_index.user_idx` is a demo rank (1..count, `num_reviews desc, user_id`, the
  order of inference's `Activity.most_active`): never a join key, a feature or an export
  column. Its `_batch_at` is the user's latest upstream `_batch_at`, so a rerun is identical.
- Timestamps are `timestamp(6)` in Iceberg tables (Athena requirement), but plain `timestamp` in
  views: Athena stores view columns as Hive types and rejects `timestamp(6)` there. Changing a mart's columns means updating
  `contracts.py` and the integration test (`on_schema_change: fail`).

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`

Athena integration: `ETL_CI_BUCKET`, `ETL_CI_WORK_GROUP`, `AWS_REGION` + credentials,
`uv run pytest tests/integration -m athena`. Add deps only with `uv add` (never edit the lock).

Infra for this component: `infrastructure/etl.tf` + the `Transform` state in `orchestration.tf`.
