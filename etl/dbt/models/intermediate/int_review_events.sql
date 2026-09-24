{{ config(unique_key='review_id') }}

-- Ledger of reviews made available to the marts: reviews of deduplicated (kept) games, each
-- stamped with the run that first released it (`_batch_at`). Every mart consumes the rows with
-- `_batch_at` greater than its own latest `_batch_at`, so each review is processed exactly once
-- per mart even when an earlier run failed half-way. Reviews of a game that is kept only later
-- (e.g. scraped after its reviews) are released in that later run.

select
    r.review_id,
    r.user_id,
    r.game_id,
    r.is_positive,
    r.reviewed_at,
    {{ batch_timestamp() }} as _batch_at
from {{ ref('int_reviews__deduplicated') }} as r
inner join {{ ref('int_games__deduplicated') }} as g on g.game_id = r.game_id
{% if is_incremental() %}
left join {{ this }} as t on t.review_id = r.review_id
where t.review_id is null
{% endif %}
