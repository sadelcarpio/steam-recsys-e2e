{{ config(unique_key='review_id') }}

-- One row per review_id, first-seen version wins (a review re-emitted by a retried scraper task
-- or re-scraped in a later run never changes a loaded row). Incremental: only scrape dates from
-- the lookback window are read, and already loaded review_ids are skipped.

with source_window as (
    select *
    from {{ ref('stg_steam__reviews') }}
    {% if is_incremental() %}
    where scrape_date >= date_add(
        'day', -{{ var('reviews_lookback_days') }},
        {{ incremental_max('scrape_date', "date '1970-01-01'") }}
    )
    {% endif %}
),

first_seen as (
    select *
    from (
        select
            *,
            row_number() over (
                partition by review_id order by scrape_date asc, updated_at desc
            ) as seen_rank
        from source_window
    )
    where seen_rank = 1
)

select
    f.review_id,
    f.user_id,
    f.game_id,
    f.is_positive,
    -- Iceberg on Athena stores microsecond timestamps only.
    cast(f.reviewed_at as timestamp(6)) as reviewed_at,
    cast(f.updated_at as timestamp(6)) as updated_at,
    f.scrape_date,
    {{ batch_timestamp() }} as _batch_at
from first_seen as f
{% if is_incremental() %}
left join {{ this }} as t on t.review_id = f.review_id
where t.review_id is null
{% endif %}
