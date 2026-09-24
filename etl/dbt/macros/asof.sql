{#-
  ASOF join helper. Emits CTEs `<prefix>_bounds`, `<prefix>_candidates`, `<prefix>_timeline`,
  `<prefix>_asof`. `<prefix>_asof` has (review_id, feature_ts): the timestamp of the latest row of
  `features` with the same `key` and `timestamp` strictly before the event's `reviewed_at`
  (null when there is none). A review therefore never sees the feature row it produced itself.
  `events` must expose review_id, reviewed_at and `key`.
  Only the feature rows that can match are read: rows at/after each key's earliest event, plus
  the single latest row before it.
-#}
{% macro asof_feature_timestamp(prefix, events, features, key) %}
{{ prefix }}_bounds as (
    select {{ key }}, min(reviewed_at) as min_ts
    from {{ events }}
    group by {{ key }}
),

{{ prefix }}_candidates as (
    select f.{{ key }}, f.timestamp as feature_ts
    from {{ features }} as f
    inner join {{ prefix }}_bounds as b on b.{{ key }} = f.{{ key }} and f.timestamp >= b.min_ts
    union all
    select f.{{ key }}, max(f.timestamp) as feature_ts
    from {{ features }} as f
    inner join {{ prefix }}_bounds as b on b.{{ key }} = f.{{ key }} and f.timestamp < b.min_ts
    group by f.{{ key }}
),

{{ prefix }}_timeline as (
    -- At equal timestamps events sort before feature rows (is_feature = 0 first): strictly before.
    select
        {{ key }},
        review_id,
        max(feature_ts) over (
            partition by {{ key }}
            order by ts, is_feature
            rows between unbounded preceding and current row
        ) as feature_ts
    from (
        select {{ key }}, feature_ts as ts, 1 as is_feature, feature_ts, cast(null as bigint) as review_id
        from {{ prefix }}_candidates
        union all
        select {{ key }}, reviewed_at as ts, 0 as is_feature, cast(null as timestamp(6)) as feature_ts, review_id
        from {{ events }}
    )
),

{{ prefix }}_asof as (
    select review_id, feature_ts
    from {{ prefix }}_timeline
    where review_id is not null
)
{% endmacro %}
