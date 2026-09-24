{{ config(unique_key='game_name_key') }}

-- Human-readable details of every current catalog game (one row per int_games__deduplicated
-- row, same key), for serving / a frontend. Details are static: they come from the scrape that
-- made the game its name's winner and are only replaced when another appid wins the name.
-- Incremental: winners merged into int_games__deduplicated since the last load.

with winners as (
    select *
    from {{ ref('int_games__deduplicated') }}
    {% if is_incremental() %}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
    {% endif %}
),

scrapes as (
    -- the winning scrape (retried scraper tasks can write it more than once)
    select
        s.game_id,
        s.game_short_description,
        s.game_header_image,
        s.game_release_date,
        s.game_price,
        row_number() over (
            partition by s.game_id
            order by s.game_short_description nulls last, s.game_header_image nulls last
        ) as copy_rank
    from {{ ref('stg_steam__games') }} as s
    inner join winners as w on w.game_id = s.game_id and w.scrape_date = s.scrape_date
)

select
    w.game_id,
    w.game_name_key,
    w.game_name,
    s.game_short_description,
    s.game_header_image,
    s.game_release_date,
    w.game_is_free,
    s.game_price,
    w.game_developers,
    w.game_publishers,
    w.game_genres,
    w.game_categories,
    w._batch_at
from winners as w
left join scrapes as s on s.game_id = w.game_id and s.copy_rank = 1
