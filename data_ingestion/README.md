# Data Ingestion

Incremental, idempotent scraping of Steam game info and reviews into
`s3://raw-steam-data-<account-id>/`, run weekly by the `steam-recsys-pipeline` Step Function.

```
EventBridge Scheduler ─► Step Functions
   1. Lambda list-partition-game-ids ─► s3://game-partitions-<acct>/{games,reviews}/<run_id>/*.json
   2. Parallel
      ├─ Distributed Map over games/<run_id>/appids-*.json  ─► ECS games-scraping   (1 task / ≤8k ids)
      ├─ Distributed Map over reviews/<run_id>/part-*.json  ─► ECS reviews-scraping (1 task / worker,
      │                                                         own public IP each)
      └─ one task, every known game                         ─► ECS tags-scraping    (best effort)
```

| Piece | Entry point | What it does |
|---|---|---|
| Lambda `list-partition-game-ids` | `steam_ingestion.list_partition_game_ids.handler.handler` | GetAppList (games only, `if_modified_since` = stored catalog cursor), registers new appids as `pending` in `game-ids-state` and re-queues changed coming-soon games (below), writes game partitions (pending appids, ≤ `GAMES_PER_TASK` each) and review partitions (all known released games, balanced by estimated request count across `NUM_REVIEW_WORKERS`). |
| ECS `games-scraping` | `python -m steam_ingestion.games_scraping` | `appdetails` + review summary for each appid → `games/<scrape-date>-<n>-<part>.parquet`; marks appids `scraped` (with `coming_soon`) / `unavailable` / retries up to `MAX_GAME_ATTEMPTS`. |
| ECS `reviews-scraping` | `python -m steam_ingestion.reviews_scraping` | Newest-first reviews per appid down to its `last_review_ts` cursor (uncapped by default: `MAX_REVIEWS_PER_GAME`), then the **backfill** of older reviews (below) → `reviews/<scrape-date>-<worker>-<part>.parquet`; advances cursors only after the rows are in S3. |

| ECS `tags-scraping` | `python -m steam_ingestion.tags_scraping` | Steam **user tags** of every `scraped` / `pending` appid (spec 8): `IStoreBrowseService/GetItems` in batches of `TAGS_BATCH_SIZE` (100), top `TAGS_PER_GAME` (20) tags with their weights, names from `IStoreService/GetTagList` → `game_tags/<scrape-date>-<part>.parquet`. The whole catalog every run (~20 min: votes change); no API key. A failure never stops the pipeline (the `Scrape` branch catches it). |

Output schemas: `src/steam_ingestion/schemas.py` (`GAMES_SCHEMA`, `REVIEWS_SCHEMA`,
`GAME_TAGS_SCHEMA`).

**Downstream note:** a crashed/retried reviews task can re-emit rows for the game it was on;
deduplicate on `rec_id` (games on `appid` + latest `scrape_date`, tags on `appid` + latest
`scraped_at`).

**Fargate Spot.** The scraping tasks run on `FARGATE_SPOT` (Terraform
`scraping_capacity_provider`, ~70% cheaper). On an interruption ECS sends SIGTERM with a 2 min
warning (`stopTimeout = 120`): the scraper stops at the next page / request (sleeps, including a
throttle cooldown, end at once), flushes its buffer, commits the finished games (a reviews game in
its backfill keeps what was fetched) and exits 143. Step Functions retries the partition (up to
10 times, waiting 5 min growing to 30 min; the tags task 4 times), which resumes from the
committed state. Up to `scrape_tolerated_failure_percentage` (Terraform, default 10%) of a Map's
partitions may still fail without failing the run (dbt and inference still run; the failed
partitions catch up next run); above that the Map aborts the rest and the run fails. Set
`scraping_capacity_provider` to `FARGATE` to go back to on-demand.

### Coming-soon games (spec 10)

A game is scraped once, so an unreleased game would keep its coming-soon details forever
(about 52k games, 29% of the catalog, in September 2026). `game-ids-state` keeps each game's
`coming_soon` flag from its last scrape, and:
- the Lambda puts a `scraped`, coming-soon game back to `pending` when GetAppList reports it as
  modified since the last run (its store page changed, usually its release), so it is scraped
  again in the same run (`rescrape_game_ids` in the result). Only coming-soon games: refreshing
  prices of every modified game is out of scope (cost);
- coming-soon games are left out of the review partitions (they have no reviews). A released
  game's first review scrape fetches its whole history, so nothing is lost.

Games scraped before spec 10 have no flag (absent = released). Seed it once from the raw games
(Athena; conditional writes, safe to re-run; run it while no pipeline execution is running). It
also re-queues coming-soon games that already look released (a past day-precise release date,
or scraped reviews), whose change happened before the Lambda watched for it:

```bash
AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_coming_soon --dry-run   # counts only
AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_coming_soon
```

### Reviews backfill (spec 6)

With the default `MAX_REVIEWS_PER_GAME=0` a new game's first scrape fetches its whole history and
each later run every review since the last one. Games whose first scrape was capped (the initial
load ran with a 2000 cap) get the rest from the backfill: each run fetches up to `BACKFILL_REVIEWS_PER_RUN` **older** reviews per game with Steam's undocumented
`start_date=1&end_date=<oldest_review_ts>&date_range_type=include` range (newest-first inside
the range, `end_date` inclusive), and moves the cursor's `oldest_review_ts` back. When the range
runs out (Steam stops returning pages, for the biggest games after a few hundred thousand
reviews) the cursor gets `backfill_complete`. A review outside the range means Steam ignored it:
that game's backfill stops for the run (forward results are kept). While any backfill is
pending, the Lambda returns `etl_full_refresh: true` and the pipeline rebuilds the marts
(backfilled reviews are older than rows already loaded).

Cursors written before the backfill have no `oldest_review_ts` and are never backfilled until
seeded, once, from the raw reviews (Athena; conditional writes, safe to re-run; run it while no
pipeline execution is running):

```bash
AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_backfill --dry-run   # counts only
AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_backfill
```

## Configuration

Pydantic settings (`steam_ingestion.config.IngestionSettings`). Precedence: env vars >
SSM `/data-ingestion/<ENV_VAR>` (only read when `USE_SSM=true`, as in AWS).

| Env var | Default | |
|---|---|---|
| `RAW_BUCKET`, `PARTITIONS_BUCKET` | – (required) | |
| `GAME_IDS_TABLE` / `REVIEWS_CURSOR_TABLE` | `game-ids-state` / `reviews-state-cursor` | |
| `STEAM_API_KEY_SECRET_ID` | `data-ingestion/steam-api-key` | Secrets Manager (plain string or `{"api_key": ...}`) |
| `STEAM_API_KEY` | unset | Local fallback, skips Secrets Manager |
| `NUM_REVIEW_WORKERS` | 10 | |
| `GAMES_PER_TASK` | 8000 | ~7 h per task at the store rate limit (2 requests per game) |
| `REQUEST_INTERVAL_SECONDS` | 1.5 | Pacing per task (≈200 req / 5 min per IP) |
| `THROTTLE_COOLDOWN_SECONDS` | 60 | Minimum wait after an HTTP 429 (or `Retry-After` if longer), so retries outlast the throttle window |
| `MAX_REVIEWS_PER_GAME` | 0 | Newest reviews per game per run (forward pass); `0` = no cap. With a cap, more new reviews than this in one run leave a gap that is not backfilled |
| `BACKFILL_REVIEWS_PER_RUN` | 100000 | Older reviews per game per run (backfill); `0` = off |
| `MAX_GAME_ATTEMPTS` | 3 | |
| `MAX_FAILURE_RATIO` | 0.2 | Task exits 1 above this share of failed games |
| `PARTITION_KEY` | – | Per-task, injected by the Distributed Map |

## Development

```bash
cd data_ingestion
uv sync --extra scraping
uv run pytest            # moto + responses, no network / AWS needed
uv run ruff check . && uv run ruff format --check .
```

Run against real AWS/Steam locally (uses your AWS profile):

```bash
export RAW_BUCKET=raw-steam-data-<acct> PARTITIONS_BUCKET=game-partitions-<acct> STEAM_API_KEY=...
uv run python -c "from steam_ingestion.list_partition_game_ids.handler import handler; print(handler({'run_id': 'local-1'}, None))"
PARTITION_KEY=reviews/local-1/part-000.json uv run python -m steam_ingestion.reviews_scraping
```

## Build & deploy

- Image: `docker build -f scraping.Dockerfile -t data-ingestion .` (both ECS tasks; the task
  definition picks the module).
- Lambda zip: `scripts/build_lambda.sh` → `build/list-partition-game-ids.zip` (no polars).
- CI: `.github/workflows/data-ingestion-ci.yml` (lint, tests, image + zip build) on changes
  under `data_ingestion/`.
- CD: run **data-ingestion CD** manually (pushes `data-ingestion:<sha>` + `:latest` to ECR and
  updates the Lambda code). Requires the infrastructure to be applied first.
