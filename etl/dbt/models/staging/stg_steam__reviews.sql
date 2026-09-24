-- Only what the feature marts need. Reviews without author, game, vote or creation time are unusable.
-- Timestamps are plain `timestamp` (UTC): Athena stores view columns as Hive types, which have no
-- timestamp(6). The Iceberg models downstream cast to timestamp(6).
select
    cast(rec_id as bigint) as review_id,
    cast(author_id as bigint) as user_id,
    cast(appid as bigint) as game_id,
    voted_up as is_positive,
    cast(from_unixtime(timestamp_created) as timestamp) as reviewed_at,
    cast(from_unixtime(coalesce(timestamp_updated, timestamp_created)) as timestamp) as updated_at,
    scrape_date
from {{ source('steam', 'reviews') }}
where rec_id is not null
    and author_id is not null
    and appid is not null
    and voted_up is not null
    and timestamp_created > 0
