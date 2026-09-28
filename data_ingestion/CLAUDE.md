# data_ingestion

Specs: `specs/1-scraping-implementation.md`, `specs/6-reviews-backfill.md` (backfill of older
reviews). Human docs: `README.md`.

## Layout

- `src/steam_ingestion/` (uv project, hatchling, py3.12)
  - `config.py`: `IngestionSettings` (env > SSM `/data-ingestion/*` when `USE_SSM=true`),
    `ScrapeTaskSettings` (`PARTITION_KEY`), `resolve_steam_api_key`
  - `models.py`: Pydantic contracts (`ListPartitionEvent/Result`, `PartitionFile`,
    `GameState`/`GameStatus`, `GameRecord`, `ReviewRecord`) and partition key regexes
  - `schemas.py`: polars schemas of the raw parquet (ETL contract). **Imports polars**, so
    never import it from Lambda code paths
  - `steam_api.py`: `SteamClient` (pacing, retries on 403/429/5xx/null body; a 429
    waits at least `throttle_cooldown` / `Retry-After`, `SteamApiError`
    never contains the URL because it carries the API key)
  - `state.py`: DynamoDB access. `game-ids-state` has one item per appid, and `appid=0` is the
    catalog cursor item. `reviews-state-cursor` holds `last_review_ts`, `total_reviews` and the
    backfill fields `oldest_review_ts` (absent = unseeded, never backfilled) / `backfill_complete`
  - `storage.py` (S3, Lambda-safe), `partitioning.py` (pure)
  - `list_partition_game_ids/handler.py`, `games_scraping/scraper.py`,
    `reviews_scraping/scraper.py` (+ `__main__.py` for `python -m`): forward pass, then the
    backward (backfill) pass through `SteamClient.iter_review_pages(until_ts=...)`
  - `shutdown.py`: SIGTERM flag + interruptible sleep (Fargate Spot interruptions)
  - `seed_backfill.py`: one-off, seeds the backfill fields of older cursors from Athena
- `scraping.Dockerfile`: one image for both ECS tasks
- `scripts/build_lambda.sh`: Lambda zip with base deps only (no `scraping` extra)

## Invariants (keep them when changing code)

- Lambda: persist new appids as `pending` **before** moving the catalog cursor, and clear
  `games/<run_id>/` and `reviews/<run_id>/` before writing, so re-running a run_id is idempotent.
- Scrapers: never overwrite output. `next_part_number` continues part numbering on retries.
- Games are marked `scraped` only after their parquet part is written.
- Review cursors are committed only for games whose rows are fully flushed. A game that was
  split across a flush keeps its old cursor (duplicates are possible, gaps are not).
- Backfill: `oldest_review_ts` only moves back to reviews that are buffered (so written with the
  cursor); `backfill_complete` only when a range walk ended before the budget. A failed backfill
  never fails the game. The Lambda's `etl_full_refresh` (Step Functions `Transform` override)
  must stay true while any backfill is pending.
- `MAX_REVIEWS_PER_GAME` stays `0` (default): a forward cap drops a scraped game's new reviews
  past it, a gap the backfill never reaches. Partition weights therefore count a scraped game's
  new reviews (`SCRAPED_GAME_NEW_REVIEWS`), not its lifetime `total_reviews`.
- SIGTERM (Fargate Spot interruption): `shutdown.Shutdown` only sets a flag; the scrapers poll it
  and the Steam client's sleeps raise `ShutdownRequested`. They then flush, commit what the
  invariants above allow (never a half-walked forward pass) and exit `EXIT_CODE` (143,
  non-zero so Step Functions retries). Never exit 0 on a stop: the partition would count as done.
- Partition keys: `games/<run_id>/appids-NNN.json`, `reviews/<run_id>/part-NNN.json`.
  The worker id in the output names comes from `NNN`.
- Steam API key only in Secrets Manager (`data-ingestion/steam-api-key`) or the local
  `STEAM_API_KEY` env. Never log it.

## Commands

`uv sync --extra scraping && uv run pytest && uv run ruff check . && uv run ruff format --check .`

Tests use moto (`mock_aws`) and `responses`. Fixtures are in `tests/conftest.py`. Every
feature needs tests.

Infra for this component: `infrastructure/data_ingestion.tf` + `infrastructure/orchestration.tf`.
