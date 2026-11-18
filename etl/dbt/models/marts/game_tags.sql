{{ config(unique_key='game_id', on_schema_change='append_new_columns') }}

-- Current Steam user tags of every catalog game (spec 8): its latest tags scrape, encoded with
-- lkp_tags, with Steam's weights in the same order (weight descending). Static per game, like
-- its genres: not time-versioned (later votes leak into older training rows, accepted).
-- Incremental: scrapes since the latest loaded scrape_date minus games_lookback_days; only
-- games whose scrape is newer than the loaded one (or new to the mart) are merged.
-- game_content_descriptors (spec 19): Steam's mature content descriptors of the same scrape
-- (3 = adult only sexual content, inference's adult filter), [] = none declared, null = scraped
-- before they were kept. Added with append_new_columns: old rows stay null until the next
-- tags scrape (every run) merges them.

with source_window as (
    select *
    from {{ ref('stg_steam__game_tags') }}
    where cardinality(game_tags) > 0
    {% if is_incremental() %}
        and scrape_date >= date_add(
            'day', -{{ var('games_lookback_days') }},
            {{ incremental_max('scrape_date', "date '1970-01-01'") }}
        )
    {% endif %}
),

latest_scrape as (
    select *
    from (
        select
            *,
            row_number() over (
                partition by game_id order by scraped_at desc, scrape_date desc
            ) as scrape_rank
        from source_window
    )
    where scrape_rank = 1
),

tag_map as (
    select map_agg(name, id) as tags from {{ ref('lkp_tags') }}
)

select
    s.game_id,
    l.game_idx,
    {{ encode_array('s.game_tags', 'm.tags') }} as game_tags,
    transform(s.game_tag_weights, w -> cast(w as double)) as game_tag_weights,
    s.game_content_descriptors,
    s.scraped_at,
    s.scrape_date,
    {{ batch_timestamp() }} as _batch_at
from latest_scrape as s
inner join {{ ref('lkp_games') }} as l on l.game_id = s.game_id
cross join tag_map as m
{% if is_incremental() %}
left join {{ this }} as t on t.game_id = s.game_id
where t.game_id is null or s.scraped_at > t.scraped_at
{% endif %}
