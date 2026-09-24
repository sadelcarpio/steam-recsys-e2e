{{ config(unique_key=['game_id', 'timestamp']) }}

-- Cumulative review counts per game: a 1970-01-01 row (no reviews yet) for every kept game, then
-- one row per second in which the game got reviews, with the counts through that second.
-- Incremental: new ledger reviews are added on top of each game's latest loaded counts. Reviews
-- arrive newest-last per game (review cursors), so this matches a full rebuild; a review older
-- than the game's latest loaded row is counted from the latest state (run with FULL_REFRESH to
-- recompute exactly).

with new_reviews as (
    select game_id, reviewed_at, is_positive, _batch_at
    from {{ ref('int_review_events') }}
    {% if is_incremental() %}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
    {% endif %}
),

per_second as (
    select
        game_id,
        reviewed_at as ts,
        count_if(is_positive) as positive,
        count_if(not is_positive) as negative,
        max(_batch_at) as _batch_at
    from new_reviews
    group by game_id, reviewed_at
),

{% if is_incremental() %}
prior as (
    select
        c.game_id,
        max_by(c.positive_reviews, c.timestamp) as positive,
        max_by(c.negative_reviews, c.timestamp) as negative
    from {{ this }} as c
    where c.game_id in (select game_id from per_second)
    group by c.game_id
),
{% endif %}

review_rows as (
    select
        s.game_id,
        s.ts,
        {% if is_incremental() %}coalesce(p.positive, 0) + {% endif %}sum(s.positive) over (
            partition by s.game_id order by s.ts rows between unbounded preceding and current row
        ) as positive,
        {% if is_incremental() %}coalesce(p.negative, 0) + {% endif %}sum(s.negative) over (
            partition by s.game_id order by s.ts rows between unbounded preceding and current row
        ) as negative,
        s._batch_at
    from per_second as s
    {% if is_incremental() %}
    left join prior as p on p.game_id = s.game_id
    {% endif %}
),

base_rows as (
    select
        g.game_id,
        timestamp '1970-01-01 00:00:00' as ts,
        cast(0 as bigint) as positive,
        cast(0 as bigint) as negative,
        {{ batch_timestamp() }} as _batch_at
    from {{ ref('int_games__deduplicated') }} as g
    {% if is_incremental() %}
    where g.game_id not in (select game_id from {{ this }})
    {% endif %}
)

select
    game_id,
    cast(ts as timestamp(6)) as timestamp,
    positive as positive_reviews,
    negative as negative_reviews,
    _batch_at
from (
    select * from review_rows
    union all
    select * from base_rows
)
