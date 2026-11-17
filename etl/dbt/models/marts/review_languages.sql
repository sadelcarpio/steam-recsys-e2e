{{ config(unique_key='review_id') }}

-- The language a review was written in (Steam API name, e.g. english, schinese), one row per
-- review_id (spec 17, A7: the retriever's user tower reads the language of the user's previous
-- review). Join on review_id to interactions for the user and time. A table of its own so no
-- existing Iceberg model changes schema. Deduplicated like int_reviews__deduplicated (first
-- seen among the copies with a language; a re-scrape never changes a loaded row). Reviews
-- without a language are left out (consumers treat a missing review as unknown).
-- Incremental: only scrape dates from the lookback window (behind this table's newest
-- scrape_date) are read, and loaded review_ids are skipped.

with source_window as (
    select review_id, language, scrape_date, updated_at
    from {{ ref('stg_steam__reviews') }}
    -- before the dedup, so incremental and full refresh keep the same copy
    where language is not null
    {% if is_incremental() %}
    and scrape_date >= date_add(
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
    f.language,
    f.scrape_date,
    {{ batch_timestamp() }} as _batch_at
from first_seen as f
{% if is_incremental() %}
left join {{ this }} as t on t.review_id = f.review_id
where t.review_id is null
{% endif %}
