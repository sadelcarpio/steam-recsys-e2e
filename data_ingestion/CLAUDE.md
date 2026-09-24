# data_ingestion

Spec: `specs/1-scraping-implementation.md`. Human docs: `README.md`.

## Layout

- `src/steam_ingestion/` (uv project, hatchling, py3.12)
  - `config.py`: `IngestionSettings` (env > SSM `/data-ingestion/*` when `USE_SSM=true`),
    `ScrapeTaskSettings` (`PARTITION_KEY`), `resolve_steam_api_key`
  - `models.py`: Pydantic contracts (`ListPartitionEvent/Result`, `PartitionFile`,
    `GameState`/`GameStatus`, `GameRecord`, `ReviewRecord`) and partition key regexes
  - `schemas.py`: polars schemas of the raw parquet (ETL contract). **Imports polars**, so
    never import it from Lambda code paths
  - `steam_api.py`: `SteamClient` (pacing, retries on 403/429/5xx/null body, `SteamApiError`
    never contains the URL because it carries the API key)
  - `state.py`: DynamoDB access. `game-ids-state` has one item per appid, and `appid=0` is the
    catalog cursor item. `reviews-state-cursor` holds `last_review_ts` and `total_reviews`
  - `storage.py` (S3, Lambda-safe), `partitioning.py` (pure)
  - `list_partition_game_ids/handler.py`, `games_scraping/scraper.py`,
    `reviews_scraping/scraper.py` (+ `__main__.py` for `python -m`)
- `scraping.Dockerfile`: one image for both ECS tasks
- `scripts/build_lambda.sh`: Lambda zip with base deps only (no `scraping` extra)

## Invariants (keep them when changing code)

- Lambda: persist new appids as `pending` **before** moving the catalog cursor, and clear
  `games/<run_id>/` and `reviews/<run_id>/` before writing, so re-running a run_id is idempotent.
- Scrapers: never overwrite output. `next_part_number` continues part numbering on retries.
- Games are marked `scraped` only after their parquet part is written.
- Review cursors are committed only for games whose rows are fully flushed. A game that was
  split across a flush keeps its old cursor (duplicates are possible, gaps are not).
- Partition keys: `games/<run_id>/appids-NNN.json`, `reviews/<run_id>/part-NNN.json`.
  The worker id in the output names comes from `NNN`.
- Steam API key only in Secrets Manager (`data-ingestion/steam-api-key`) or the local
  `STEAM_API_KEY` env. Never log it.

## Commands

`uv sync --extra scraping && uv run pytest && uv run ruff check . && uv run ruff format --check .`

Tests use moto (`mock_aws`) and `responses`. Fixtures are in `tests/conftest.py`. Every
feature needs tests.

Infra for this component: `infrastructure/data_ingestion.tf` + `infrastructure/orchestration.tf`.
