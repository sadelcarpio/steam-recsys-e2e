-- sync_all_columns: dropping a column (games_reviewed_positive, 2026-11) is an Iceberg
-- metadata change (ALTER TABLE DROP COLUMN), not a rebuild.
{{ config(unique_key='review_id', on_schema_change='sync_all_columns') }}

-- Training examples: one row per review of a kept game, with the game features as of strictly
-- before the review (ASOF join on game_features), so the label (is_positive) never leaks into
-- its own features. User histories are not stored: training and inference take each user's
-- last N positive reviews before a row from these rows (any N, per model).

with batch as (
    select review_id, user_id, game_id, is_positive, reviewed_at, _batch_at
    from {{ ref('int_review_events') }}
    {% if is_incremental() %}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
    {% endif %}
),

{{ asof_feature_timestamp('games', 'batch', ref('game_features'), 'game_id') }}

select
    b.review_id,
    b.reviewed_at as timestamp,
    b.user_id,
    b.game_id,
    g.game_idx,
    b.is_positive,
    g.game_name,
    g.game_is_free,
    g.game_developers,
    g.game_publishers,
    g.game_genres,
    g.game_categories,
    g.game_reviews_ratio,
    b._batch_at
from batch as b
left join games_asof as ga on ga.review_id = b.review_id
left join {{ ref('game_features') }} as g
    on g.game_id = b.game_id and g.timestamp = ga.feature_ts
