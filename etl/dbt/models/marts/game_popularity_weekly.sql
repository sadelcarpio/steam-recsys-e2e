{{ config(unique_key=['game_id', 'week_start']) }}

-- Positive / total reviews per game and ISO week (week_start = Monday 00:00 UTC of the review's
-- timestamp), from interactions (spec 9). The one definition of game popularity: ranker features
-- (lifetime / last 30 / last 90 days as of t), popularity lists, the online game table. Point in
-- time by construction: a consumer at time t reads only the weeks before t's week.
-- Incremental: every (game, week) touched by an interactions row newer than the watermark is
-- recounted in full from interactions, so late reviews (the backfill writes old weeks) and
-- re-reviewed rows land in the right week. _batch_at = the newest interactions row counted.

{% if is_incremental() %}
with touched as (
    select distinct
        game_id,
        cast(date_trunc('week', timestamp) as date) as week_start
    from {{ ref('interactions') }}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
)
{% endif %}

select
    i.game_id,
    arbitrary(i.game_idx) as game_idx,
    cast(date_trunc('week', i.timestamp) as date) as week_start,
    count_if(i.is_positive) as positives,
    count(*) as reviews,
    max(i._batch_at) as _batch_at
from {{ ref('interactions') }} as i
{% if is_incremental() %}
inner join touched as t
    on t.game_id = i.game_id
    and t.week_start = cast(date_trunc('week', i.timestamp) as date)
{% endif %}
group by i.game_id, cast(date_trunc('week', i.timestamp) as date)
