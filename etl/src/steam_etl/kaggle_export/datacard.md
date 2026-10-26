Steam user reviews, game details and Steam user tags, collected from the Steam Web API and
store by an open-source recommender system pipeline: $repo_url.

**Snapshot:** $snapshot_date (UTC). New versions are published from the pipeline, so row counts
grow from one version to the next.

## Files

| File | Rows | One row per |
|---|---|---|
| `reviews*.parquet` | $reviews | review |
| `games.parquet` | $games | game |
| `game_tags.parquet` | $tags | game with user tags |

### reviews

`review_id`, `user_id`, `game_id` (Steam appid), `is_positive` (recommended or not),
`review` (text), `language`, `reviewed_at` / `updated_at` (UTC), `playtime_forever` /
`playtime_last_two_weeks` / `playtime_at_review` (minutes), `num_games_owned`, `num_reviews`,
`votes_up`, `votes_funny`, `weighted_vote_score`, `comment_count`, `steam_purchase`,
`received_for_free`, `written_during_early_access`, `primarily_steam_deck`.

A review appears once: the version first collected is kept.

### games

`game_id`, `game_name`, `game_short_description`, `game_header_image`, `game_release_date`,
`game_is_free`, `game_price`, `game_developers`, `game_publishers`, `game_genres`,
`game_categories`, `game_coming_soon` (not released when last collected).

### game_tags

`game_id`, `tag_names` / `tag_weights` (Steam user tags and their vote weights, by weight
descending), `scraped_at` (UTC).

## Users and privacy

`user_id` is **not** a Steam id. It is a keyed hash (HMAC-SHA256, truncated to 63 bits) of the
reviewer's SteamID64. It is stable across versions, so a user's reviews can be grouped, and it
can't be reversed or recomputed without the key, which is never published.

Steam reviews are public, though: searching a review's text on Steam can lead to its author's
profile. The hash only prevents joining this data with Steam ids directly. Don't use this
dataset to identify or contact reviewers.

## License and terms

- Review texts belong to their authors. Game names, descriptions, images and tags belong to
  Valve and the games' publishers. This dataset claims no rights over them ("Data files ©
  Original Authors").
- Collected from Steam's public Web API and store pages, for research and education.
- To have content removed, open an issue at $repo_url/issues.
