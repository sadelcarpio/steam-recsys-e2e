# Data Ingestion

Incremental, idempotent scraping of Steam game info and reviews into
`s3://raw-steam-data-<account-id>/`, run weekly by the `steam-recsys-pipeline` Step Function.

```
EventBridge Scheduler ─► Step Functions
   1. Lambda list-partition-game-ids ─► s3://game-partitions-<acct>/{games,reviews}/<run_id>/*.json
   2. Parallel
      ├─ Distributed Map over games/<run_id>/appids-*.json  ─► ECS games-scraping   (1 task / ≤8k ids)
      └─ Distributed Map over reviews/<run_id>/part-*.json  ─► ECS reviews-scraping (1 task / worker,
                                                                own public IP each)
```

| Piece | Entry point | What it does |
|---|---|---|
| Lambda `list-partition-game-ids` | `steam_ingestion.list_partition_game_ids.handler.handler` | GetAppList (games only, `if_modified_since` = stored catalog cursor), registers new appids as `pending` in `game-ids-state`, writes game partitions (pending appids, ≤ `GAMES_PER_TASK` each) and review partitions (all known games, balanced by estimated request count across `NUM_REVIEW_WORKERS`). |
| ECS `games-scraping` | `python -m steam_ingestion.games_scraping` | `appdetails` + review summary for each appid → `games/<scrape-date>-<n>-<part>.parquet`; marks appids `scraped` / `unavailable` / retries up to `MAX_GAME_ATTEMPTS`. |
| ECS `reviews-scraping` | `python -m steam_ingestion.reviews_scraping` | Newest-first reviews per appid down to its `last_review_ts` cursor → `reviews/<scrape-date>-<worker>-<part>.parquet`; advances cursors only after the rows are in S3. |

Output schemas: `src/steam_ingestion/schemas.py` (`GAMES_SCHEMA`, `REVIEWS_SCHEMA`).

**Downstream note:** a crashed/retried reviews task can re-emit rows for the game it was on;
deduplicate on `rec_id` (games on `appid` + latest `scrape_date`).

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
| `MAX_REVIEWS_PER_GAME` | 2000 | Newest reviews per game per run; `0` = full history |
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
