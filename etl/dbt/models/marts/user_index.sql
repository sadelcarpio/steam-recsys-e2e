{{ config(materialized='table') }}

-- Demo user index (spec 13): every user of `interactions`, user_idx 1 = most reviews ...
-- count(*) = least (ties: lowest user_id), the same order inference uses to pick MAX_USERS and
-- the reranked users. Rebuilt every run: deterministic for the same data, but a user's
-- user_idx moves as new reviews arrive, so it is never a join key or stored with anything
-- persistent. _batch_at comes from the user's rows, so a rerun without new data is identical.

with activity as (
    select
        user_id,
        count(*) as num_reviews,
        count_if(is_positive) as num_positive,
        max(timestamp) as last_reviewed_at,
        max(_batch_at) as _batch_at
    from {{ ref('interactions') }}
    group by user_id
)

select
    row_number() over (order by num_reviews desc, user_id) as user_idx,
    user_id,
    num_reviews,
    num_positive,
    last_reviewed_at,
    _batch_at
from activity
