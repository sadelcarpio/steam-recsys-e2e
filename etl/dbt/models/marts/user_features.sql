{{ config(unique_key=['user_id', 'timestamp']) }}

{% set n = var('user_history_length') %}

-- Time-versioned user features, one row per (user, second of a positive review):
-- games_reviewed_positive = the user's last {{ n }} positively reviewed games (dense game_idx),
-- most recent first, right-padded with 0, including that second's reviews.
-- Incremental: every user with a new positive review is recomputed from their full history, so
-- late-arriving reviews are placed correctly.

with
{% if is_incremental() %}
affected_users as (
    select distinct user_id
    from {{ ref('int_review_events') }}
    where is_positive
        and _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
),
{% endif %}

positives as (
    select e.user_id, e.review_id, e.reviewed_at, l.game_idx, e._batch_at
    from {{ ref('int_review_events') }} as e
    inner join {{ ref('lkp_games') }} as l on l.game_id = e.game_id
    {% if is_incremental() %}
    inner join affected_users as a on a.user_id = e.user_id
    {% endif %}
    where e.is_positive
),

windowed as (
    select
        user_id,
        reviewed_at,
        array_agg(game_idx) over (
            partition by user_id
            order by reviewed_at, review_id
            rows between {{ n - 1 }} preceding and current row
        ) as recent_oldest_first,
        -- the last review of a second carries the history through that second
        row_number() over (
            partition by user_id, reviewed_at order by review_id desc
        ) as rank_in_second,
        max(_batch_at) over (partition by user_id) as _batch_at
    from positives
)

select
    user_id,
    reviewed_at as timestamp,
    {{ pad_ids('reverse(recent_oldest_first)', n) }} as games_reviewed_positive,
    _batch_at
from windowed
where rank_in_second = 1
