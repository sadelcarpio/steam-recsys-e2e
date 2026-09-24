{#-
  Append-only dense id vocabulary. New values get max(id) + 1, +2, ... in sorted order, so ids
  never change once assigned. Ids 0 (padding) and 1 (OOV) are reserved: the first real id is 2.
  `values_sql` must select one column named `value`.
-#}
{% macro dense_id_lookup(values_sql, id_column, value_column) %}
with candidates as (
    select distinct value from ({{ values_sql }})
    where value is not null
),

new_values as (
    select c.value
    from candidates as c
    {% if is_incremental() %}
    left join {{ this }} as t on t.{{ value_column }} = c.value
    where t.{{ value_column }} is null
    {% endif %}
)

select
    {% if is_incremental() %}{{ incremental_max(id_column, oov_id()) }}{% else %}cast({{ oov_id() }} as bigint){% endif %}
        + row_number() over (order by value) as {{ id_column }},
    value as {{ value_column }},
    {{ batch_timestamp() }} as _batch_at
from new_values
{% endmacro %}

{% macro attribute_lookup(array_column) %}
{% set values_sql %}
    select value
    from {{ ref('int_games__deduplicated') }}
    cross join unnest({{ array_column }}) as u (value)
{% endset %}
{{ dense_id_lookup(values_sql, 'id', 'name') }}
{% endmacro %}
