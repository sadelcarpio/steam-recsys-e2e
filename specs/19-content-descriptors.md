## Spec 19: Steam content descriptors for the adult filter

### Problem

The adult filter (`inference/src/steam_inference/adult.py`) relied on heuristics: Steam's
"Sexual Content" / "Nudity" genres, explicit words in the name, and user tags. The tag rules were
too broad. Checked against the marts on 2026-11-17, they hid about 10.7k games (about 4% of
reviews), including:

- `Nudity` / `Sexual Content` among the top 5 tags: Cyberpunk 2077, GTA V Enhanced, GTA San
  Andreas DE, Stellar Blade, Like a Dragon: Infinite Wealth, Atomic Heart.
- `NSFW` / `Hentai` anywhere in the tags (trolling votes far down the list): Marvel Rivals,
  KovaaK's, Blue Archive.

The goal is to show mature mainstream games and still hide games that are actually
pornographic.

### Signal: Steam's content descriptors (verified 2026-11-17)

Developers answer Steam's mature content survey, and the store exposes the answers as content
descriptor ids:

| id | meaning |
|----|---------|
| 1 | Some Nudity or Sexual Content |
| 2 | Frequent Violence or Gore |
| 3 | Adult Only Sexual Content |
| 4 | Frequent Nudity or Sexual Content |
| 5 | General Mature Content |

Descriptor 3 is the line Steam itself draws: the store hides these games unless the user opts in.

| Game | Descriptors |
|------|-------------|
| Cyberpunk 2077, GTA V Enhanced, Stellar Blade | 1, 2, 5 |
| GTA San Andreas DE | 2, 5 |
| Marvel Rivals, Apex Legends, Civilization VII | none |
| HuniePop, Mirror | 1, 3, 4, 5 |
| NEKOPARA Vol. 1 (the safe-for-work release) | 1, 5 |

`IStoreBrowseService/GetItems`, the call the tags scraper already makes (spec 8), returns them as
`content_descriptorids`. The field is absent, not `[]`, when a game declares none. Collecting them
costs no extra request, and they are refreshed for every known game on every run. `appdetails`
also has them (`content_descriptors.ids`), but only new games are scraped there, so using it
would mean re-scraping the whole catalog.

### Changes

- **Ingestion** (`tags-scraping`): `SteamClient.get_game_tags` returns a `StoreItem` (tags plus
  sorted, de-duplicated descriptors). `GameTagsRecord` and the raw parquet `game_tags/*` get
  `content_descriptors: list[int]`, which is `[]` when the game declares none. As before, a game
  without tags writes no row.
- **Glue** (`infrastructure/etl.tf`): the raw `game_tags` table gets `content_descriptors
  array<bigint>`. Parquet columns are matched by name, so older files read as null.
- **dbt**: `stg_steam__game_tags.content_descriptors` becomes `game_content_descriptors`
  (sorted, distinct) in the `game_tags` mart:
  - `[]` means the game declares none; null means it was scraped before this spec.
  - The mart uses `on_schema_change='append_new_columns'`: existing rows stay null until the
    next tags scrape merges them, which happens on every run.
  - `GameTagsRow` contract: `game_content_descriptors: list[int] | None`.
- **Inference**:
  - `load_games` reads the column into `Games.content_descriptors`: an int64 bitmask per catalog
    row, -1 when unknown. It is None when the mart predates the column, so an image deployed
    before the ETL change still runs.
  - `adult_mask` adds `has_descriptor(3)` to the existing rules.
- **Tag rule** (in the same change): the top-5 and "anywhere" tag rules are replaced by the share
  of tag votes that the adult tags hold. A game is adult at 17% or more, or at 10% or more when
  an adult tag is its most voted tag. Mature mainstream games sit at 7–16% (Cyberpunk "Nudity"
  is 3rd, GTA V "Sexual Content" 2nd), trolling tags at about 1%, and adult-only games at 20–66%.
  This rule covers games without descriptors and the ones whose descriptors understate them
  (NEKOPARA).

No retraining is needed. The retriever and the ranker are trained on every game, and the filter
applies only to what inference publishes.

### Rollout

1. Merge: the CD applies Terraform (the Glue column) and deploys the data_ingestion, etl and
   inference images.
2. The next scheduled pipeline run: tags-scraping writes the descriptors, dbt adds and fills the
   column, and inference applies the new rule. A manual inference run is not enough before
   that, because the descriptors are only collected on a pipeline run. The vote-share rule
   alone already brings back Cyberpunk, GTA and Marvel Rivals.

### Not in scope

- A "mature" toggle in the frontend (descriptors 1, 4 and 5 without 3). The bitmask is already
  in `Games` if serving or the search index needs it later.
- `scripts/publish_search_index.py` keeps the genre and name rules only. It can only remove
  games from a catalog that inference already built with the full rule.
