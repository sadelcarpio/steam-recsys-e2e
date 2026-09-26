# Inference

Weekly batch recommendations. It runs as a SageMaker Processing job (`steam-recsys-infer-*`,
`ml.t3.xlarge` by default), the last step (`Infer`) of the `steam-recsys-pipeline` Step
Function, after `Transform`. The step is skipped
(`NoChampion`) while no model has been promoted.

```
S3 models/champion/  ─► two-tower model (steam_training)
steam_marts (Iceberg, pyiceberg)
  user_features  latest row per user   ─► user tower ─┐
  game_features  latest row per game   ─► item tower ─┴► exact top K per user (reviewed games excluded)
  interactions   reviewed games, review counts, recent positives        │
                                                                    ▼
                           1000 most active users (>= 6 reviews) ─► Bedrock LLM rerank + explanations
                                                                    │
                                                                    ▼
                              DynamoDB game-explainable-recommendations (one item per user, changed ones only,
                                                                        + "__popular__" fallback item)
  game_details   text, image URL (new games only) ─► DynamoDB game-details (read by serving)
  item embeddings ─► S3 serving/online/ catalog.npz + manifest.json (serving's online endpoint,
                    with the model's models/<id>/user_tower.npz from training)
  catalog names    ─► S3 serving/search/games.json (the frontend's in-browser game search)
```

1. **Features.** The latest `user_features` / `game_features` row is each user's / game's
   current state. `interactions` supplies each user's reviewed games (excluded from the
   recommendations) and review counts. All reads are
   pinned to the snapshot current at the start of the run.
2. **Retrieval.** Every game is embedded once. Users are scored against all games in chunks
   (exact brute force, no ANN), and the top `TOP_K` (30) are kept. Games newer than the model
   are ranked from their content features (their ids map to OOV).
3. **Reranking.** The users with at least `RERANK_MIN_REVIEWS` (6) reviews, most active first
   and at most `RERANK_MAX_USERS` (1000), are sent to Bedrock (Converse API) with their last
   `games_reviewed_positive` (the last 5 liked games, the user tower's input) and their K
   candidates. Each game is one line: name | genres | developers | free or paid | positive
   review share | Steam short description (mart `game_details`, cut at
   `RERANK_DESCRIPTION_CHARS`). The model must call a `submit_ranking` tool
   that returns every candidate once, best first, and explains the first `EXPLAIN_TOP_N` (5).
   Invalid answers are repaired: unknown or repeated numbers are dropped and missing
   candidates are appended. An answer with no usable tool call (e.g. stop reason
   `malformed_tool_use`, about 0.3% of users with Nova 2 Lite) is retried up to twice. A user
   that still fails keeps the retrieval order.
4. **Output.** One item per user who has `user_features`. Only the users whose
   recommendations changed are written. The run first scans the table for each stored
   `content_hash` (reading only `user_id` and `content_hash`). An item is rewritten only when
   the hash of its new content differs: the model, the rerank flag, and the ordered games with
   their names and explanations. Scores are left out of the hash, because the weekly
   `reviews_ratio` updates move every score a little. Stored users that no longer get
   recommendations are deleted, but only on full runs (`MAX_USERS=0`). A new champion changes
   `model_id`, so it rewrites every user. The run logs `written` / `unchanged` / `deleted`.
   Users with only negative reviews have no `user_features` row and get no item.
5. **Popularity fallback.** The reserved item `user_id = "__popular__"` holds the `TOP_K`
   catalog games with the most positive reviews in the last `POPULAR_WINDOW_DAYS` (90) before
   the newest review (`model_id` `popularity`, never reranked). Serving returns it to users
   without an item. It goes through the same change check, and it is never deleted.
6. **Game details.** The mart `game_details` (name, short description, header image URL,
   release date, price, developer / publisher / genre / category names) is loaded into the
   DynamoDB table `game-details`, one item per game. Details are static, so this is
   insert-only: the run scans the stored `game_id`s and writes only the missing games. The
   first run loads the catalog (about 50k items, about $0.06), and later runs write only new
   games. Descriptions are cleaned to plain text (tags dropped, HTML entities decoded). The
   sync is skipped with a warning while the mart does not exist, and it also runs only when a
   model exists.
7. **Online catalog.** Every run writes the catalog side of serving's online model to
   `s3://model-artifacts-<acct>/serving/online/catalog.npz` (`contracts.ONLINE_BUNDLE_ARRAYS`,
   about 14 MB for about 50k games). It holds the item embeddings scored in step 2, plus each
   catalog row's `game_id`, `game_idx` and name. Then it writes `manifest.json`
   (`OnlineBundleManifest`), which:
   - pins the catalog's S3 version id;
   - names the model's numpy user tower, `models/<model_id>/user_tower.npz`. Training writes
     that file with every model, and it never changes.

   The catalog is rewritten every run because the item embeddings move with the weekly
   `reviews_ratio` and new games. Old versions expire through the bucket's 90-day
   noncurrent-version rule. When the model has no `user_tower.npz` (it was saved before the
   export existed), nothing is published and the previous manifest stays; run
   `python -m steam_training export --model-id champion` (training README).
8. **Search index.** With the online catalog (same games, same condition), the run writes
   `s3://model-artifacts-<acct>/serving/search/games.json` (`contracts.SearchIndex`): every
   catalog game as `[appid, name, reviews]`, most reviewed first, where `reviews` counts the
   reviews loaded by this run. Stored gzipped with `Content-Encoding: gzip` (about 2-3 MB for
   ~180k games); the frontend downloads it through CloudFront and searches it in the browser.
   To publish it without a run (from the catalog already in S3, counts from one Athena query):
   `uv run python scripts/publish_search_index.py [--dry-run]`.

## Output contract (`src/steam_inference/contracts.py`)

```json
{
  "user_id": "76561198027267313",
  "recommendations": [
    {"game_id": 63910, "name": "King's Bounty: Crossworlds", "score": 0.4127,
     "explanation": "As a fan of strategy and RPGs like Dungeons 2, ..."},
    {"game_id": 203350, "name": "King's Bounty: Warriors of the North", "score": 0.4343}
  ],
  "model_id": "local-test",
  "generated_at": "2026-09-24T12:40:00+00:00",
  "reranked": true,
  "rerank_model": "us.amazon.nova-2-lite-v1:0",
  "content_hash": "9f2c0e5d7a1b4c8e0f3a6b9d2c5e8f1a"
}
```

```json
{
  "game_id": 63910,
  "name": "King's Bounty: Crossworlds",
  "short_description": "Crossworlds is a stand-alone add-on ...",
  "header_image": "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/63910/header.jpg",
  "release_date": "24 Sep, 2010",
  "is_free": false,
  "price": 9.99,
  "developers": ["Katauri Interactive"], "publishers": ["1C Entertainment"],
  "genres": ["RPG", "Strategy"], "categories": ["Single-player"]
}
```

The second item is a `game-details` item (`GameDetails`). Null fields are left out.
Serving (`serving/src/steam_serving/contracts.py`) reads both tables.
`tests/test_serving_contract.py` runs the pipeline and then parses and serves every written
item with the serving package, so a contract change that breaks serving fails here.

`recommendations` is ordered best first: the LLM order when `reranked`, the model order
otherwise. `score` is the two-tower cosine similarity, so it is not monotonic after reranking.
Only the first `EXPLAIN_TOP_N` entries of a reranked list have an `explanation`. `score` and
`generated_at` are as of the last write: an unchanged list keeps them. `user_id` is a string,
because Steam ids exceed JavaScript's safe integers.

## LLM choice and cost

Bedrock has no free tier. Measured on the real marts with the same prompt, each reranked user
costs about 1.8k input tokens and 0.4k output tokens without the short descriptions; at
`RERANK_DESCRIPTION_CHARS=200` they add about 50 tokens per game, roughly 1.8k more input
tokens for 35 games (estimate):

| Model (`BEDROCK_MODEL_ID`) | Quality of the explanations | Relative cost |
|---|---|---|
| `amazon.nova-lite-v1:0` | Misattributes games and calls candidates "liked" | lowest |
| **`us.amazon.nova-2-lite-v1:0`** (default) | Grounded in the liked list, consistent | ~ 1x |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Most specific (cites review %, developers) | ~ 2.5x |

At the default of 1000 users per weekly run, that is about 1.8M input and 0.4M output tokens.
Check the current per-token prices on the Bedrock pricing page. The job logs the exact usage
(`bedrock usage: ... tokens`). To change the model, set the Terraform variable
`inference_bedrock_model_id`, which updates both the SSM parameter and the IAM permission.

DynamoDB, on-demand, per run:
- **Change check:** a scan of about 1.4M items of 1.7 KB on average costs about 0.7M read
  units, roughly $0.10.
- **Writes:** about 2 write units per changed item, at $0.625 per million. Rewriting every user
  (the first run, or a new champion) costs about $1.80. On normal weeks only the changed users
  are paid for.

## Configuration

Pydantic settings (`steam_inference.config.InferenceSettings`). Precedence: env vars > SSM
`/inference/<ENV_VAR>` (read only when `USE_SSM=true`, as in AWS).

| Env var | Default | |
|---|---|---|
| `MODEL_ARTIFACTS_BUCKET` | – (required) | `model-artifacts-<acct>` |
| `MODEL_ID` | `champion` | Or any `models/<id>/` (manual runs) |
| `GLUE_DATABASE` | `steam_marts` | |
| `RECOMMENDATIONS_TABLE` | `game-explainable-recommendations` | |
| `GAME_DETAILS_TABLE` / `SYNC_GAME_DETAILS` | `game-details` / true | Insert-only game details |
| `POPULAR_WINDOW_DAYS` | 90 | Window of the popularity fallback |
| `ONLINE_BUNDLE_ENABLED` / `ONLINE_BUNDLE_PREFIX` | true / `serving/online` | Online catalog + manifest (dry runs: `online/` next to `OUTPUT_PATH`) |
| `SEARCH_INDEX_KEY` | `serving/search/games.json` | Frontend search index, written with the online catalog (dry runs: `online/games.json.gz`) |
| `OUTPUT_PATH` | – | Write JSON lines to this local file instead of DynamoDB (details go to `game-details.jsonl` next to it) |
| `TOP_K` | 30 | Candidates kept and written per user |
| `MAX_USERS` | 0 (all) | Only the N most active users (local runs) |
| `RERANK_ENABLED` | true | |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` | Any Converse model with tool use |
| `RERANK_MIN_REVIEWS` / `RERANK_MAX_USERS` | 6 / 1000 | Who gets reranked |
| `EXPLAIN_TOP_N` | 5 | Explained recommendations per reranked user |
| `RERANK_DESCRIPTION_CHARS` | 200 | Short description per prompt game, cut at a word boundary; `0` = none (the mart `game_details` is then not read) |
| `RERANK_CONCURRENCY` / `WRITE_CONCURRENCY` | 32 / 8 | Parallel Bedrock calls (~1000 req/min, half the 2000 RPM quota) / DynamoDB scan segments and writers |
| `USER_BATCH_SIZE` / `ITEM_BATCH_SIZE` / `NUM_THREADS` | 1024 / 4096 / 0 | Scoring |

## Development

```bash
cd inference
uv sync                  # CPU torch; steam-training (and, for tests, steam-serving) are editable path deps
uv run pytest            # synthetic marts, moto S3 + DynamoDB, fake LLM: no AWS
uv run ruff check . && uv run ruff format --check .
```

Dry run on the real marts and champion, with your AWS profile. It writes a local file and makes
a handful of Bedrock calls:

```bash
MODEL_ARTIFACTS_BUCKET=model-artifacts-<acct> OUTPUT_PATH=/tmp/recs.jsonl \
  MAX_USERS=200 RERANK_MAX_USERS=5 uv run python -m steam_inference
```

Build the image from the **repository root**: `docker build -f inference/Dockerfile -t inference .`

## CI/CD

- `inference CI` (PRs and main; `inference/**`, training and serving sources): ruff, tests,
  image build.
- `inference CD` (manual): tests, then pushes `inference:<sha>` + `:latest` (the job runs
  `:latest`). With `run_now`, it also runs the job once and waits for it (runbook:
  `docs/deployment.md` step 10).
