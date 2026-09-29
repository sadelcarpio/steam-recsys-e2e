## Spec 10: Coming-soon games

### Problem

A game is scraped once and never again. On 2026-09-29, 51,756 of ~180k scraped games had
`coming_soon = true` ("Coming soon", "To be announced", "Q4 2026", ...), 50,142 of them in
`game_details`. Once released they keep their coming-soon details forever (no price, old
description), they are recommendable and searchable although unreleased, and the reviews scraper
polls all of them every run for nothing.

### Behaviour

**Ingestion**

- `game-ids-state` stores `coming_soon` (bool) with every scrape (`mark_scraped`).
- Re-scrape trigger: the Lambda's GetAppList call (`if_modified_since` = catalog cursor) also
  returns known apps whose store data changed. A known `scraped` game with `coming_soon = true`
  in that list goes back to `pending` (attempts reset) and is scraped again in the same run.
  Persisted before the catalog cursor moves (a crash can never lose the trigger). Only
  coming-soon games: re-scraping every modified game (prices) is out of scope (cost).
- Reviews: coming-soon games are left out of the review partitions (no reviews exist). Nothing is
  lost: a released game's first review scrape fetches its whole history.
- `ListPartitionResult.rescrape_game_ids`: re-queued games of the run.
- One-off `python -m steam_ingestion.seed_coming_soon` fills `coming_soon` of existing items from
  the latest raw scrape per appid (Athena), and re-queues coming-soon games that already show
  signs of a release (a day-precise release date in the past, or scraped reviews): their
  modification happened before this feature and will not be reported again.

**ETL**

- `stg_steam__games.game_coming_soon`, carried by `int_games__deduplicated` (latest scrape) to
  `game_details`.
- `int_games__deduplicated` merges an existing winner again when its `game_coming_soon` changed
  (the release), so `game_details` gets the released scrape (price, description, date). Other
  re-scrapes of a winner are still not merged (details stay static).
- A re-scraped game keeps its stored `game_name_key` (a rename on release would otherwise leave
  the appid under two names).

**Inference**

- Unreleased games (`game_details.game_coming_soon`) join the `excluded` mask with adult games:
  never recommended, not in the popular item, the online catalog, the search index or prompts.
- DynamoDB `game-details` holds released games only: unreleased ones are not inserted and stored
  ones are deleted, so a release inserts fresh details (the table stays insert-only otherwise).

### Out of scope

Refreshing prices / descriptions of released games; "upcoming games" as a product feature;
training changes (unreleased games have no interactions).

### Tests / docs / deploy

Unit tests for the state, the Lambda trigger and review exclusion, the seed script, the dbt
model changes (contracts, integration fixtures), the inference mask and the details sync. READMEs
and CLAUDE.md of `data_ingestion`, `etl`, `inference`; `docs/deployment.md`: ETL full refresh
(new columns, `on_schema_change: fail`), then the seed script.
