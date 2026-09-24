{{ config(unique_key='review_id') }}

-- Training examples: one row per review of a kept game, with the user and game features as of
-- strictly before the review (ASOF join on user_features / game_features), so the label
-- (is_positive) never leaks into its own features.

with batch as (
    select review_id, user_id, game_id, is_positive, reviewed_at, _batch_at
    from {{ ref('int_review_events') }}
    {% if is_incremental() %}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
    {% endif %}
),

{{ asof_feature_timestamp('users', 'batch', ref('user_features'), 'user_id') }},

{{ asof_feature_timestamp('games', 'batch', ref('game_features'), 'game_id') }}

select
    b.review_id,
    b.reviewed_at as timestamp,
    b.user_id,
    b.game_id,
    g.game_idx,
    b.is_positive,
    coalesce(
        u.games_reviewed_positive, {{ pad_ids('null', var('user_history_length')) }}
    ) as games_reviewed_positive,
    g.game_name,
    g.game_is_free,
    g.game_developers,
    g.game_publishers,
    g.game_genres,
    g.game_categories,
    g.game_reviews_ratio,
    g.game_review_score,
    b._batch_at
from batch as b
left join users_asof as ua on ua.review_id = b.review_id
left join {{ ref('user_features') }} as u
    on u.user_id = b.user_id and u.timestamp = ua.feature_ts
left join games_asof as ga on ga.review_id = b.review_id
left join {{ ref('game_features') }} as g
    on g.game_id = b.game_id and g.timestamp = ga.feature_ts
