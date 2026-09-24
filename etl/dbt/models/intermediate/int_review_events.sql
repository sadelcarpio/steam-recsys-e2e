{{ config(unique_key='review_id') }}

-- Ledger of reviews made available to the marts: reviews of deduplicated (kept) games, each
-- stamped with the run that first released it (`_batch_at`). Every mart consumes the rows with
-- `_batch_at` greater than its own latest `_batch_at`, so each review is processed exactly once
-- per mart even when an earlier run failed half-way. Reviews of a game that is kept only later
-- (e.g. scraped after its reviews) are released in that later run.
--
-- One review per (user, game): the first one wins. Repeats get a new review_id (double
-- submissions seconds apart, or delete + rewrite), so the review_id dedup upstream keeps them.
-- Incremental: a review whose pair is already in the ledger is never released.

with candidates as (
    select r.review_id, r.user_id, r.game_id, r.is_positive, r.reviewed_at
    from {{ ref('int_reviews__deduplicated') }} as r
    inner join {{ ref('int_games__deduplicated') }} as g on g.game_id = r.game_id
    {% if is_incremental() %}
    left join {{ this }} as t on t.review_id = r.review_id
    where t.review_id is null
    {% endif %}
),

first_per_pair as (
    select
        *,
        row_number() over (
            partition by user_id, game_id order by reviewed_at, review_id
        ) as pair_rank
    from candidates
)

select
    c.review_id,
    c.user_id,
    c.game_id,
    c.is_positive,
    c.reviewed_at,
    {{ batch_timestamp() }} as _batch_at
from first_per_pair as c
{% if is_incremental() %}
left join {{ this }} as t on t.user_id = c.user_id and t.game_id = c.game_id
{% endif %}
where c.pair_rank = 1
{% if is_incremental() %}
    and t.review_id is null
{% endif %}
