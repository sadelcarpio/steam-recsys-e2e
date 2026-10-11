-- Only what the feature marts need. Reviews without author, game, vote or creation time are unusable.
-- user_id is the scraper's keyed hash of the author's SteamID64 (spec 13): no raw id is stored.
-- Timestamps are plain `timestamp` (UTC): Athena stores view columns as Hive types, which have no
-- timestamp(6). The Iceberg models downstream cast to timestamp(6).
select
    cast(rec_id as bigint) as review_id,
    cast(user_id as bigint) as user_id,
    cast(appid as bigint) as game_id,
    voted_up as is_positive,
    cast(from_unixtime(timestamp_created) as timestamp) as reviewed_at,
    cast(from_unixtime(coalesce(timestamp_updated, timestamp_created)) as timestamp) as updated_at,
    scrape_date
from {{ source('steam', 'reviews') }}
where rec_id is not null
    and user_id is not null
    and appid is not null
    and voted_up is not null
    and timestamp_created > 0
