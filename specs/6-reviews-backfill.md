## Spec 6: Reviews backfill (older reviews past the per-game cap)

### Problem

`reviews-scraping` keeps the newest `MAX_REVIEWS_PER_GAME` (2000) reviews of a game on its first
scrape and only fetches newer ones afterwards, so older reviews are never scraped. Every game
therefore covers a different time window: a few weeks for a very popular game, years for a niche
one. With a temporal train / validation split this removes the most popular games from training
(on the first full backfill ~40% of the validation interactions were for games with no training
row: Portal 2, Slime Rancher, ...).

### Steam API behaviour (verified 2026-09-25 on appid 620)

`appreviews` accepts undocumented `start_date`, `end_date` (unix seconds) and
`date_range_type=include`. With `filter=recent` the reviews come newest-first **inside the range**
and the cursor pages through it; `end_date` is inclusive; a range with no reviews returns an empty
page. `start_date=0` is treated as "no range" (use `1`). In practice Steam stops returning pages
after a few hundred thousand reviews for the biggest games, so the walk ends when pages run out,
not when `total_reviews` is reached.

### Behaviour

Per game and run, after the existing forward pass (new reviews since `last_review_ts`):

- **Backward pass**: pages `[1, oldest_review_ts]` newest-first, up to `BACKFILL_REVIEWS_PER_RUN`
  reviews (default 100000, `0` disables), then moves `oldest_review_ts` back to the oldest review
  fetched. An empty page / repeated cursor marks the game `backfill_complete`. Reviews at the
  boundary second may be fetched twice (the ETL dedupes on `review_id`).
- A returned review newer than `end_date` means Steam ignored the range: the backward pass stops
  for that game (logged, the game is not marked complete, forward results are kept).
- **First scrape of a game**: when the forward walk ends before the cap, the game's full history
  is scraped (`backfill_complete`); when the cap truncated it, `oldest_review_ts` is its oldest
  fetched review and later runs backfill from there.
- Cursor state (`reviews-state-cursor`) gains `oldest_review_ts` (null = unknown, legacy cursor:
  never backfilled until seeded) and `backfill_complete`. Cursors are still committed only once
  the game's rows are flushed.

### Seeding existing cursors (one-off)

`python -m steam_ingestion.seed_backfill` reads `min(timestamp_created)` and the review count
per appid from the raw reviews (Athena, Glue `steam_raw.reviews`) and sets `oldest_review_ts` /
`backfill_complete` (count >= Steam's `total_reviews`) on cursors that have neither. Idempotent
(conditional writes), `--dry-run` prints the counts. Run it while no pipeline execution runs.

### Partitioning

The Lambda adds `BACKFILL_REVIEWS_PER_RUN / 100` requests to the weight of every game with a
pending backfill, and reports `backfill_game_ids` and `etl_full_refresh` in its result.

### ETL

Backfilled reviews are older than rows already in the marts. `int_game_review_counts` (running
totals) and the as-of histories in `interactions` assume new reviews are newer, so while any
backfill is pending the pipeline's `Transform` runs dbt with `FULL_REFRESH=true` (Step Functions
container override from `etl_full_refresh`). Lookups are append-only and never rebuilt.

### Forward cap off by default

A capped forward pass of an already scraped game drops its new reviews past the cap: a game with
more new reviews than the cap in one week would leave a gap the backfill never reaches (it only
walks back from `oldest_review_ts`). `MAX_REVIEWS_PER_GAME` therefore defaults to `0` (code and
Terraform): every run fetches all new reviews, and a game new to the catalog is scraped in full
on its first run (`backfill_complete`). The cap only served the initial load (tens of thousands of
first scrapes); the games it truncated are finished by the backfill in resumable chunks.

The Lambda weighs an already scraped game by at most 2000 new reviews per run
(`SCRAPED_GAME_NEW_REVIEWS`) instead of its lifetime `total_reviews`, and a new game by its full
history.

Remaining limitation: a game that joins the catalog with a very large history is scraped in one
long first pass (about 15 s per 1000 reviews); a task failure mid-game restarts that game.

### Tests / docs / infra

Scraper, API client, state, partitioning, Lambda and seeding tests; SSM parameter +
Terraform variable `backfill_reviews_per_run`; `Transform` override; README / CLAUDE.md /
`docs/deployment.md` (seeding step).

## Addendum: backfill cap and once per run (2026-09-30)

- `BACKFILL_MAX_REVIEWS_PER_GAME` (default 500000, `0` = off): the backfill stops once a game has
  that many reviews stored; the forward pass is not affected. The cursor counts them in
  `stored_reviews` (forward + backward rows committed with it); cursors from before the counter
  are seeded once with `python -m steam_ingestion.seed_stored_reviews` (pending backfills only,
  Athena counts), and are not capped until then. The Lambda weighs a capped game's backfill as
  `min(BACKFILL_REVIEWS_PER_RUN, cap - stored_reviews)` and does not count it as pending.
- Once per run: the cursor records the run id of its last backward pass (`backfill_run_id`). A
  retried task (Fargate Spot interruption) skips the backward pass of games already backfilled
  in the run, instead of spending another `BACKFILL_REVIEWS_PER_RUN` on each.
