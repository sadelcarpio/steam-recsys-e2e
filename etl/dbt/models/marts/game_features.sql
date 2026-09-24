{{ config(unique_key=['game_id', 'timestamp']) }}

{% set alpha = var('reviews_ratio_prior') %}

-- Time-versioned game features, one row per int_game_review_counts row (a 1970-01-01 row, then
-- one row per second with reviews). The latest row per game is its current state.
-- game_reviews_ratio is Laplace-smoothed: (positive + {{ alpha }}) / (positive + negative + 2 * {{ alpha }}),
-- 0.5 before the first review.
-- Steam's review_score is deliberately not a feature: it is scraped once, so older reviews would
-- see a score computed from later reviews (leakage). int_games__deduplicated still uses it for
-- the name dedup.

with lookup_maps as (
    select
        (select map_agg(name, id) from {{ ref('lkp_developers') }}) as developers,
        (select map_agg(name, id) from {{ ref('lkp_publishers') }}) as publishers,
        (select map_agg(name, id) from {{ ref('lkp_genres') }}) as genres,
        (select map_agg(name, id) from {{ ref('lkp_categories') }}) as categories
),

games as (
    select
        g.game_id,
        l.game_idx,
        g.game_name,
        g.game_is_free,
        {{ encode_array('g.game_developers', 'm.developers') }} as game_developers,
        {{ encode_array('g.game_publishers', 'm.publishers') }} as game_publishers,
        {{ encode_array('g.game_genres', 'm.genres') }} as game_genres,
        {{ encode_array('g.game_categories', 'm.categories') }} as game_categories
    from {{ ref('int_games__deduplicated') }} as g
    inner join {{ ref('lkp_games') }} as l on l.game_id = g.game_id
    cross join lookup_maps as m
),

counts as (
    select *
    from {{ ref('int_game_review_counts') }}
    {% if is_incremental() %}
    where _batch_at > {{ incremental_max('_batch_at', "timestamp '1970-01-01 00:00:00'") }}
    {% endif %}
)

select
    c.game_id,
    g.game_idx,
    c.timestamp,
    g.game_name,
    g.game_is_free,
    g.game_developers,
    g.game_publishers,
    g.game_genres,
    g.game_categories,
    (c.positive_reviews + cast({{ alpha }} as double))
        / (c.positive_reviews + c.negative_reviews + 2 * cast({{ alpha }} as double))
        as game_reviews_ratio,
    c._batch_at
from counts as c
inner join games as g on g.game_id = c.game_id
