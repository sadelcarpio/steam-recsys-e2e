-- Steam user tag name -> id (spec 8), over the tags of catalog games (like the other lookups).
{% set values_sql %}
    select value
    from {{ ref('stg_steam__game_tags') }}
    cross join unnest(game_tags) as u (value)
    where game_id in (select game_id from {{ ref('int_games__deduplicated') }})
{% endset %}
{{ dense_id_lookup(values_sql, 'id', 'name') }}
