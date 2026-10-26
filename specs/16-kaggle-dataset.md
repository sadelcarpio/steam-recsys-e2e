## Spec 16: Kaggle dataset (pseudonymous, published from the pipeline without blocking it)

### Problem

The public Kaggle dataset `sergiodelcarpio/steam-reviews-100m` was published by hand, before
spec 13, with raw SteamID64s. Kaggle can't delete single versions, only the whole dataset.
The data in AWS now only has the pseudonymous `user_id` (spec 13), is deduplicated, and grows
every week.

Goal: a new dataset, refreshed automatically from the pipeline, that never delays or fails a
run. Once it's published and checked, the old dataset is deleted.

### What gets published

Dataset `sergiodelcarpio/steam-reviews` (SSM `/etl/KAGGLE_DATASET`; the slug is decided at the
first publish, and the old slug isn't reused). Three Parquet tables, as flat files (Kaggle
gets no directories):

| Files | Source | Rows |
|---|---|---|
| `reviews-NNN.parquet` (`ROWS_PER_FILE`, 10M rows each; `reviews.parquet` when one file) | raw `steam_raw.reviews`, deduplicated with the rule of `int_reviews__deduplicated` (one row per `rec_id`, the first scrape date wins, then the latest `timestamp_updated`) | ~118M |
| `games.parquet` | mart `steam_marts.game_details` (one row per game, `game_coming_soon` marks unreleased ones: name, short description, header image, release date, price, developers, publishers, genres, categories; without the internal `game_name_key`) | estimate ~180k |
| `game_tags.parquet` | the latest raw `steam_raw.game_tags` scrape of each game in `games` (tag **names** and weights; the mart `game_tags` only has encoded ids) | estimate ~120k |

Row counts are estimates (the Oct 11 inference log); the data card states the real ones.

`reviews` columns: `review_id` (was `rec_id`), `user_id` (the spec 13 hash, bigint), `game_id`,
`is_positive`, `review` (text), `language`, `reviewed_at` / `updated_at` (UTC timestamps),
`playtime_forever` / `playtime_last_two_weeks` / `playtime_at_review` (minutes),
`num_games_owned`, `num_reviews`, `votes_up`, `votes_funny`, `weighted_vote_score`,
`comment_count`, `steam_purchase`, `received_for_free`, `written_during_early_access`,
`primarily_steam_deck`.

Left out:
- `scrape_date` and every other scraper bookkeeping column;
- `last_played`, an exact per-user timestamp that adds linkage risk and no modelling value;
- any `user_idx` (spec 13: demo numbers change every run and are never exported);
- the HMAC key, which never leaves Secrets Manager.

`int_reviews__deduplicated` keeps no text or playtime, so the export doesn't read it. It
applies the same dedup rule to raw, and the task checks that its review count equals
`count(*)` of `int_reviews__deduplicated`. A mismatch fails the export (not the pipeline).

**Data card** (the Kaggle description, generated from a template in the repo):
- sources and the snapshot date;
- the schema;
- how `user_id` was made: a keyed hash, stable across versions, not reversible without the key;
- the privacy caveat: the review text is public on Steam, so a review can be matched to its
  author's profile by searching its text, and the hash only prevents joins on SteamID;
- the license (the same as the old dataset; check its page before the first publish);
- a link to the repository.

### When it runs: after `Transform`, fire and forget

New state `PublishKaggle` between `Transform` and `CheckChampion`:

```hcl
PublishKaggle = {
  Type     = "Task"
  Resource = "arn:aws:states:::ecs:runTask"   # no .sync: returns once the task is started
  Parameters = { ...kaggle-export task, RUN_ID = $$.Execution.Name ... }
  Catch      = [{ ErrorEquals = ["States.ALL"], ResultPath = null, Next = "CheckChampion" }]
  ResultPath = null
  Next       = "CheckChampion"
}
```

- `runTask` without `.sync` only waits for ECS to accept the task (seconds). The export then
  runs in parallel with inference, and the execution can end before it does.
- The `Catch` makes even a failed launch (capacity, IAM) a no-op for the pipeline. No retry:
  the next run tries again.
- An export failure only shows up in the task's logs (`/ecs/etl`, stream prefix
  `kaggle-export`) and its exit code. Spec 11's alarms can watch it later.

**Frequency.** Every Kaggle version re-uploads every file (there are no incremental versions):
about 17 GB, roughly $1.5 of internet egress per publish. The task decides on its own:
- it reads `s3://processed-steam-data-<acct>/exports/kaggle-state/last.json` (`KaggleExportState`:
  `published_at`, `run_id`, `version_note`, `review_count`);
- it exits 0 without doing anything when the last publish is younger than
  `/etl/KAGGLE_MIN_DAYS_BETWEEN` (default 28, about monthly; 0 = every run).

The state machine stays the same whatever the frequency. Changing the frequency is an SSM
edit.

### The task: `python -m steam_etl.kaggle_export`

ETL component (it already has the Athena workgroup and Glue access). Steps:

1. Read the settings (`KaggleExportSettings`, pydantic, env > SSM `/etl/*`) and the state
   file; skip when it isn't due, or when `KAGGLE_EXPORT_ENABLED=false`.
2. Athena `UNLOAD (...) TO 's3://processed-steam-data-<acct>/exports/kaggle/<run_id>/<table>/'
   WITH (format = 'PARQUET', compression = 'ZSTD')`, one per table, in the `steam-recsys-etl`
   workgroup. The SQL lives in `etl/src/steam_etl/kaggle_export/sql/*.sql` and is checked by
   unit tests and the Athena integration test. Scanning about 17 GB of raw costs about $0.09.
3. Download the parts one at a time to the task's disk (`/tmp/kaggle/<run_id>/`) and rewrite
   them as the flat files above (pyarrow, streamed in batches), counting rows. Then check the
   counts (reviews against `int_reviews__deduplicated`; both game tables non-empty) before
   anything reaches Kaggle.
4. Write `dataset-metadata.json`: `id`, `title`, `subtitle`, `description` (the data card,
   `kaggle_export/datacard.md`, with the snapshot date and row counts), `licenses`
   (`KAGGLE_LICENSE`, Terraform `kaggle_license`, default `other`). No keywords: Kaggle only
   accepts existing tags.
5. The token comes from Secrets Manager `etl/kaggle-api-token` and is passed to the Kaggle
   client as `KAGGLE_API_TOKEN` in the process environment only. It's never logged, and
   errors are logged without request details.
   The `kaggle` Python API (2.x), with `convert_to_csv=False` (its default converts uploads to
   CSV):
   - the dataset exists: `dataset_create_version(dir, note)`;
   - it doesn't: `dataset_create_new(dir, public=False)`, **private**. Making it public is a
     manual, one-time step after checking it.
6. Poll `dataset_status` until a version newer than the previous one is `ready` (`error` fails;
   time out after `KAGGLE_STATUS_TIMEOUT_MINUTES`, default 60, and exit 1). Then write `last.json`, and delete `exports/kaggle/<run_id>/`. A
   7-day lifecycle rule on `exports/kaggle/` also expires leftovers.

Exit codes: 0 when published or skipped, 1 on any failure (nothing is half-published: Kaggle
only creates the version once the upload completes).

The dependencies are `kaggle` and `pyarrow`, added with `uv add` (never by editing the lock).

### Infrastructure (`etl.tf`, `orchestration.tf`)

- **Secret** `etl/kaggle-api-token`: a placeholder value with `ignore_changes`. The real value
  is set out of band (runbook).
- **SSM** `/etl/EXPORT_S3_PREFIX`, `/etl/EXPORT_STATE_S3_URI`, `/etl/KAGGLE_DATASET`
  (`var.kaggle_dataset`), `/etl/KAGGLE_TOKEN_SECRET_ID`, `/etl/KAGGLE_LICENSE`
  (`var.kaggle_license`). The runtime switches `/etl/KAGGLE_EXPORT_ENABLED` (created `false`)
  and `/etl/KAGGLE_MIN_DAYS_BETWEEN` (created `28`) have `ignore_changes` on their value, so
  `aws ssm put-parameter` changes stick.
- **Task definition** `kaggle-export`:
  - image `etl:${var.etl_image_tag}`, command `python -m steam_etl.kaggle_export`;
  - 1 vCPU / 4 GB, `ephemeral_storage { size_in_gib = 60 }`;
  - public subnet with `assignPublicIp` (no NAT, as the scrapers);
  - logs to `/ecs/etl`, prefix `kaggle-export`.
- **Role** `steam-recsys-kaggle-export` (inline policy):
  - Athena query and get on the `etl` workgroup;
  - Glue read on `steam_raw`, `steam_intermediate` and `steam_marts`;
  - S3 read of `raw-steam-data-*/reviews/*`, `raw-steam-data-*/game_tags/*` and the Iceberg
    prefix, and write on `athena-results/*` (query results are written with the caller's
    permissions);
  - S3 read, write and delete on `processed-steam-data-*/exports/kaggle/*` and
    `exports/kaggle-state/*`;
  - `secretsmanager:GetSecretValue` on the token only;
  - `ssm:GetParametersByPath` on `/etl/`.
- **Lifecycle**: expire objects under `exports/kaggle/` after 7 days. `last.json` lives under
  `exports/kaggle-state/`, outside that prefix, so it's kept.
- **State machine**: the `PublishKaggle` state, plus `ecs:RunTask` on the task definition and
  `iam:PassRole` on its roles for `aws_iam_role.pipeline`.

### Tests

- `etl` unit tests (no AWS, no Kaggle):
  - the UNLOAD SQL renders, and its dedup rule matches `int_reviews__deduplicated` (same
    partition and order: a test compares the two `row_number()` clauses);
  - the reviews query selects exactly the published columns (no `scrape_date`, no
    `last_played`, no `author_id`, no `user_idx`);
  - the skip logic (no state file, a recent publish, `MIN_DAYS_BETWEEN = 0`, disabled);
  - `dataset-metadata.json` (the id from settings, the data card with the counts);
  - a count mismatch fails before any upload;
  - create vs version, against a fake Kaggle client;
  - `last.json` is written only after `ready`;
  - the token never appears in logs (caplog);
  - S3 parts with moto.
- Athena integration (CI, `-m athena`): the UNLOAD of the reviews query on the fixtures
  returns one row per review, the first scrape wins, and the count matches
  `int_reviews__deduplicated`.
- `infrastructure`: `terraform validate`.

### Docs

- `etl/README.md` and `CLAUDE.md`: the export, its settings, and the frequency rule.
- `infrastructure/CLAUDE.md`: the state list gains `PublishKaggle`, the only fire-and-forget
  state.
- Root `CLAUDE.md`: control flow (3b) and data flow (a new "Public dataset" line).
- `docs/deployment.md`: setting the token (below) and the first publish.

### Rollout

1. Merge. *Infrastructure CD* (secret, SSM, task definition, role, state), then *etl CD* (the
   image with the export). With `KAGGLE_EXPORT_ENABLED=false`, the new state starts a task
   that exits at once.
2. Set the token from your own terminal; it never passes through a chat or CI log:
   ```bash
   set -a; . ./.env; set +a
   aws secretsmanager put-secret-value --profile admin --secret-id etl/kaggle-api-token \
     --secret-string "$KAGGLE_API_TOKEN"   # the variable's name in .env
   aws ssm put-parameter --profile admin --overwrite --name /etl/KAGGLE_EXPORT_ENABLED --value true
   ```
3. First publish without waiting for Thursday: run the `kaggle-export` task by hand
   (`aws ecs run-task`, as in the spec 13 migration). It creates the dataset as **private**.
4. Check it on Kaggle:
   - the files and row counts;
   - no 17-digit `user_id` starting with `7656119` (a SteamID64);
   - the data card.

   Then make it public (Settings → Visibility).
5. Delete the old dataset (irreversible; every version, votes, discussions and notebook
   links go with it):
   ```bash
   kaggle datasets delete sergiodelcarpio/steam-reviews-100m
   ```
   Check that its page is gone, then post the new link wherever the old one was shared.

### Out of scope

- Exporting the marts (`interactions`, `user_features`): they are model-specific features.
  The cleaned tables above are what a Kaggle user wants to feature-engineer from.
- Coarsening timestamps or dropping the review text (decided: the text is included).
- Uploading only the new reviews: Kaggle has no incremental versions.
- A CloudWatch alarm on export failures (spec 11).

### Implementation notes (2026-10-25)

- Code: `etl/src/steam_etl/kaggle_export/` (task `python -m steam_etl.kaggle_export`).
  Terraform: `etl.tf` (secret, SSM, role, task definition, `expire-kaggle-exports` lifecycle
  rule), `orchestration.tf` (`PublishKaggle`, pipeline RunTask / PassRole), `variables.tf`
  (`kaggle_dataset`, `kaggle_license`).
- The intermediate review table has no text or playtime, so the export reads raw with the
  intermediate dedup rule (checked by a unit test and by a count match at run time).
- `game_tags` comes from the latest raw scrape (tag names), not the mart (encoded ids).
- Not verified locally: `terraform validate` (the sandbox stops provider plugins; CI runs
  it), the Athena integration test (CI), and a real Kaggle upload (first manual publish).
