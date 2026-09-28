## Spec 8: Steam user tags as item features

### Problem

The item tower only sees developers, publishers, genres and categories from `appdetails`. Genres
are too coarse to tell games apart ("Action" / "Adventure" is on most AAA games), so the model
can't learn the finer tastes players share ("Psychological Horror", "Story Rich", "Noir",
"Souls-like"). Steam's user tags capture those, weighted by how many players applied them, but
`appdetails` does not return them.

### Steam API (verified 2026-09-28)

- `GET https://api.steampowered.com/IStoreBrowseService/GetItems/v1/?input_json=<json>` with
  `{"ids": [{"appid": N}, ...], "context": {"language": "english", "country_code": "US"},
  "data_request": {"include_tag_count": 20}}` returns `response.store_items[]` with
  `tags: [{tagid, weight}]`, the top tags by weight. No API key. Up to 200 ids per call work
  (500 fails: the URL gets too long); unknown or removed apps come back with `success != 1`.
  40 calls at ~1.3/s were not throttled.
- `GET https://api.steampowered.com/IStoreService/GetTagList/v1/?language=english` returns the
  names of the ~450 tags (`tags: [{tagid, name}]`).
- Undocumented (the Steam store's own front end uses it), so responses are validated and a
  failed tags run never blocks the pipeline.

### Ingestion: `tags-scraping` (new ECS task, same image)

- One Fargate (Spot) task per pipeline run, a third branch of the `Scrape` Parallel state (no
  partitions: one task covers the catalog in about 20 minutes).
- Games: every appid in `game-ids-state` with status `scraped` or `pending` (pending games are
  real apps from GetAppList, so games scraped in this same run get their tags too). Tags are
  refreshed for the whole catalog every run, since votes change over time.
- Calls GetTagList once, then GetItems in batches of `TAGS_BATCH_SIZE` (100) with
  `TAGS_PER_GAME` (20) tags each, paced by `TAGS_REQUEST_INTERVAL_SECONDS` (0.5 s: a separate
  host from the throttled store endpoints).
- Output: `s3://raw-steam-data-*/game_tags/<scrape-date>-<part>.parquet`, one row per game with
  at least one tag: `appid`, `tag_ids`, `tag_names`, `tag_weights` (aligned lists, by weight
  descending), `scraped_at` (unix seconds), `scrape_date`. Flushed every `TAGS_FLUSH_EVERY`
  games. A tag id missing from GetTagList is dropped.
- Retries: a SIGTERM flushes and exits 143 (Step Functions retries the task, which re-scrapes
  everything; duplicates are deduplicated by the ETL). A batch that fails after the client's
  retries is skipped; the task fails when more than `MAX_FAILURE_RATIO` of the batches failed.
  The `Scrape` branch catches a final failure, so the pipeline goes on with the previous tags.

### ETL

- Raw Glue table `steam_raw.game_tags` (Terraform, mirrors the parquet schema).
- `stg_steam__game_tags` (view): names trimmed, pairs with a null / empty name dropped.
- `lkp_tags` (append-only, like the other lookups): tag name → dense id (0 padding, 1 OOV).
- Mart `game_tags` (incremental Iceberg, merged on `game_id`): the latest scrape of each game in
  `lkp_games`: `game_id`, `game_idx`, `game_tags` (lkp ids, by weight descending),
  `game_tag_weights` (Steam's weights as double, same order), `scrape_date`, `_batch_at`.
  Incremental window: scrapes since `max(scrape_date) - games_lookback_days`.
- Tags are the current state of a game, like its genres: a static feature, not time-versioned
  (the small leakage of later votes is accepted, as with genres).

### Training

- `VocabSizes.tags` (`None` for models trained without tags, so the current champion still
  loads: no architecture version bump). `USE_GAME_TAGS` (default `true`) reads `lkp_tags` and
  the `game_tags` mart; the catalog joins each game's tags by `game_idx` (no tags: empty bag).
- Item tower: one more `EmbeddingBag` (`attribute_embedding_dim`, mode `sum`) with per-sample
  weights = the game's tag weights normalized to sum 1 (a weighted mean of its tag embeddings),
  concatenated with the other attributes before the MLP.
- The user tower is unchanged (game ids only), so `user_tower.npz` and serving are unchanged.
- Promotion loads tags when the candidate or the champion uses them.

### Inference

- The catalog joins the tags from the `game_tags` mart whenever it exists (a model trained with
  tags and no mart fails loudly). The online catalog item embeddings include them, so the
  serving Lambda needs no change. Inference must be deployed before a model with tags is
  promoted: older images reject the new `vocab.tags` field in its metadata.
- Rerank prompt: the game description lists its top 5 tags (when loaded).
- Adult filter: also a game with the `NSFW` or `Hentai` tag, or `Sexual Content` / `Nudity` among
  its top 5 tags (lower-ranked, those tags also appear on mainstream games such as RPGs).

### Rollout

infrastructure apply (task definition, Glue table, state machine) → data-ingestion CD → ETL
CD → a pipeline run (scrapes tags, builds `game_tags`) → training CD (the new model uses tags)
→ promote. Until a model with tags is promoted, inference keeps serving the current champion.

### Out of scope

Tags in the frontend cards / `game-details` table, tags on the user tower side.
