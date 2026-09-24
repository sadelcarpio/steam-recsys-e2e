# ETL

dbt on Athena that turns the raw scraper parquet (`s3://raw-steam-data-<acct>/{games,reviews}/`)
into Iceberg feature tables for the two-tower model. It runs as the ECS task `dbt`, the
`Transform` step of the `steam-recsys-pipeline` Step Function (after `Scrape`).

```
steam_raw (Glue external, Terraform)  ─► steam_staging (views)
   games, reviews                          stg_steam__games, stg_steam__reviews
                                      ─► steam_intermediate (Iceberg)
                                           int_games__deduplicated   one game per name
                                           int_reviews__deduplicated one row per review_id
                                           int_review_events         ledger of reviews released to the marts
                                           int_game_review_counts    cumulative +/- counts per game and second
                                      ─► steam_marts (Iceberg)
                                           lkp_games, lkp_developers, lkp_publishers,
                                           lkp_genres, lkp_categories  dense id vocabularies
                                           game_features, user_features  time-versioned features
                                           interactions                  training examples (ASOF join)
                                           game_details                  human-readable details (serving)
```

## Marts

| Table | Grain | Columns |
|---|---|---|
| `lkp_*` | one value | `id` (`game_idx` for games), `name` (`game_id`), `_batch_at` |
| `game_features` | (`game_id`, `timestamp`): a 1970-01-01 row (no reviews yet), then one row per second in which the game got reviews | `game_idx`, `game_name`, `game_is_free`, `game_developers` / `_publishers` / `_genres` / `_categories` (id arrays), `game_reviews_ratio` = Laplace-smoothed (pos + α) / (pos + neg + 2α), α = var `reviews_ratio_prior` (1), 0.5 without reviews |
| `user_features` | (`user_id`, `timestamp`) of each positive review | `games_reviewed_positive`: last 5 positively reviewed `game_idx`, most recent first, right-padded with 0 |
| `game_details` | one current catalog game (the same rows as `int_games__deduplicated`) | `game_id`, `game_name_key`, `game_name`, `game_short_description`, `game_header_image` (URL), `game_release_date` (text as shown on Steam), `game_is_free`, `game_price`, `game_developers` / `_publishers` / `_genres` / `_categories` (**names**). Not a model feature: inference loads it into DynamoDB `game-details` for serving. |
| `interactions` | `review_id` | `timestamp`, `user_id`, `game_id`, `is_positive` (label) + all user and game features **as of strictly before** the review |

The latest `game_features` / `user_features` row per key is the current state (for inference).

Steam's `review_score` (0-9 summary bucket) is **not** a feature: it is scraped once per game, so
a review from years earlier would see a score computed from later reviews. `game_reviews_ratio`
carries the same signal correctly as of each review. The score is only used as the name-dedup
tie-break in `int_games__deduplicated`.
Row contracts: `src/steam_etl/contracts.py`.

**Reserved ids** in every vocabulary: `0` = padding (fills fixed-length lists, never a value),
`1` = out-of-vocabulary (a value without an id). Real ids start at `2`.

### Cleaning rules
- Games: latest scrape per appid; `type = game` (or unknown) with a non-empty name; then one game
  per case-insensitive name, keeping the highest `review_score` (ties: more recommendations,
  lowest appid). Arrays are trimmed, blanks dropped, deduplicated.
- Reviews: one per `review_id`, **first-seen version wins** (later edits are ignored). Reviews
  without author, game, vote or creation time are dropped. Then **one review per (user, game), the
  first**: repeats get a new `review_id` (double submissions seconds apart, delete + rewrite;
  ~55 pairs in 2.2M reviews). Reviews of games that are not kept
  (lost the name dedup, not scraped yet) stay out of the marts until their game is kept.

## Incremental + idempotent

- `int_*`: read only scrape dates from the last loaded one minus a lookback (3 days), skip
  already loaded keys, merge on the key.
- `int_review_events` stamps each newly released review with the run start (`_batch_at`). Every
  mart takes ledger rows newer than its own max `_batch_at`, so a review is processed exactly
  once per mart, even after a run that failed half-way.
- `int_game_review_counts` adds new counts on top of each game's latest row (raw counts stay in
  the intermediate layer; the marts only carry the smoothed ratio). `user_features` recomputes
  the users with new positive reviews from their full history. `interactions` ASOF-joins only
  the new reviews.
- `lkp_*` are append-only (`full_refresh: false`): new values get the next ids, existing ids
  never change.
- Rerunning with no new raw data writes nothing. The Athena integration test checks this, and that
  incremental loads equal a full rebuild.
- Known approximation: a review older than its game's latest loaded row (it would need to arrive
  late) is counted on top of the latest state. `FULL_REFRESH=true` recomputes exactly (lookups
  are kept).

## ID conversion: at the transformation layer or as a training artifact?

Decision: the transformation layer assigns ids (`lkp_*`). Training stores each vocabulary size
(`max(id) + 1`) and the Iceberg snapshot id it read, next to the model artifacts.

Rough sizes: about 100k games after filtering, tens of thousands of developers and publishers,
about 30 genres and about 60 categories. Every vocabulary is at most about 10^5 rows (a few MB),
so where the mapping happens does not matter for cost. The choice rests on these points:
- **Stable ids.** Append-only lookups keep a value on the same id across weekly runs, so a
  deployed model still maps old items the same way. Anything added after training gets an
  id >= that model's vocabulary size, and inference maps it to OOV (1).
- **One mapping.** Training and batch inference read the same tables, so there is no artifact to
  keep in sync, and Iceberg time travel reproduces the vocabulary of any training run.
- **Embedding-ready marts.** Features are already integers, and history lists are already padded.
- User ids are not encoded. There are millions of users with sparse signal, so the user tower is
  built from `games_reviewed_positive`, not from a user-id embedding.

A training-artifact vocabulary would tie the ids to the model instead. Its drawbacks are that
ids change on every retrain and inference needs extra mapping code. It becomes worthwhile only
if vocabularies are pruned by frequency per model.

## Configuration

Pydantic settings (`steam_etl.config.EtlSettings`). Precedence: env vars > SSM `/etl/<ENV_VAR>`
(read only when `USE_SSM=true`, as in AWS).

| Env var | Default | |
|---|---|---|
| `ATHENA_WORK_GROUP` | – (required) | `steam-recsys-etl` |
| `ATHENA_S3_STAGING_DIR` | – (required) | `s3://processed-steam-data-<acct>/athena-results/` |
| `ICEBERG_S3_DATA_DIR` | – (required) | `s3://processed-steam-data-<acct>/iceberg/` |
| `DBT_SCHEMA` | `steam` | Glue databases `<schema>_raw/_staging/_intermediate/_marts` |
| `AWS_REGION` | `us-east-1` | |
| `DBT_THREADS` | 4 | |
| `FULL_REFRESH` | false | Rebuild everything except the lookups |
| `ICEBERG_MAINTENANCE` | true | OPTIMIZE + VACUUM each incremental table after writing it (off in the integration test) |
| `REVIEWS_LOOKBACK_DAYS` / `GAMES_LOOKBACK_DAYS` | 3 | Re-read scrape dates before the latest loaded |

## Development

```bash
cd etl
uv sync
uv run pytest            # unit tests + dbt parse, no AWS (Athena test skips)
uv run ruff check . && uv run ruff format --check .
```

Athena integration test (what the `etl CI` workflow runs, with the `steam-recsys-etl-ci` role;
the `ETL_CI_*` values are Terraform outputs, see `docs/deployment.md` steps 3-4).
It builds everything in throwaway `ci_*` Glue databases and deletes them afterwards:

```bash
AWS_REGION=us-east-1 ETL_CI_BUCKET=etl-ci-<acct> ETL_CI_WORK_GROUP=steam-recsys-etl-ci \
  uv run pytest tests/integration -m athena -v -s   # ~5 min; ETL_CI_KEEP=true keeps the databases
```

Run dbt against prod from a workstation (your AWS profile):

```bash
export ATHENA_WORK_GROUP=steam-recsys-etl ATHENA_S3_STAGING_DIR=s3://processed-steam-data-<acct>/athena-results/ \
       ICEBERG_S3_DATA_DIR=s3://processed-steam-data-<acct>/iceberg/
uv run python -m steam_etl
```

Full rebuild on AWS: run the `dbt` task with a `FULL_REFRESH=true` container override.

## CI/CD

- `etl CI` (PRs and main, `etl/**`): ruff, unit tests, the Athena integration test (OIDC role
  `steam-recsys-etl-ci`) and the image build.
- `etl CD` (manual): tests, then pushes `etl:<sha>` and `etl:latest` to ECR. The pipeline's
  task definition runs `:latest`.
