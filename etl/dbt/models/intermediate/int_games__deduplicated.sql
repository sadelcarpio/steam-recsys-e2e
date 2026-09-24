{{ config(unique_key='game_name_key') }}

-- One game per (case-insensitive) name: the appid with the highest review score wins
-- (ties: more recommendations, then lowest appid). Incremental: games scraped since the last
-- load compete with the current winner of their name, and only new or replaced winners are merged.

with source_window as (
    select *
    from {{ ref('stg_steam__games') }}
    where game_name is not null
        and coalesce(game_type, 'game') = 'game'
    {% if is_incremental() %}
        and scrape_date >= date_add(
            'day', -{{ var('games_lookback_days') }},
            {{ incremental_max('scrape_date', "date '1970-01-01'") }}
        )
    {% endif %}
),

latest_scrape as (
    select
        game_id,
        lower(game_name) as game_name_key,
        game_name,
        game_is_free,
        game_developers,
        game_publishers,
        game_genres,
        game_categories,
        game_review_score,
        game_recommendations,
        scrape_date
    from (
        select
            *,
            row_number() over (partition by game_id order by scrape_date desc) as scrape_rank
        from source_window
    )
    where scrape_rank = 1
),

candidates as (
    select * from latest_scrape
    {% if is_incremental() %}
    union all
    select
        game_id,
        game_name_key,
        game_name,
        game_is_free,
        game_developers,
        game_publishers,
        game_genres,
        game_categories,
        game_review_score,
        game_recommendations,
        scrape_date
    from {{ this }}
    where game_name_key in (select game_name_key from latest_scrape)
    {% endif %}
),

winners as (
    select *
    from (
        select
            *,
            row_number() over (
                partition by game_name_key
                order by game_review_score desc nulls last,
                    game_recommendations desc nulls last,
                    game_id asc
            ) as name_rank
        from candidates
    )
    where name_rank = 1
)

select
    w.game_id,
    w.game_name_key,
    w.game_name,
    w.game_is_free,
    w.game_developers,
    w.game_publishers,
    w.game_genres,
    w.game_categories,
    w.game_review_score,
    w.game_recommendations,
    w.scrape_date,
    {{ batch_timestamp() }} as _batch_at
from winners as w
{% if is_incremental() %}
left join {{ this }} as t
    on t.game_name_key = w.game_name_key
    and t.game_id = w.game_id
where t.game_id is null
{% endif %}
