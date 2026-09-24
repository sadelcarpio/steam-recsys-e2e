{#- Run start time, stamped on every row a run writes (`_batch_at`). Marts use it as their watermark. -#}
{% macro batch_timestamp() -%}
  cast('{{ run_started_at.strftime("%Y-%m-%d %H:%M:%S.%f") }}' as timestamp(6))
{%- endmacro %}

{#- Trimmed, non-empty, distinct strings, order preserved; null array -> empty array. -#}
{% macro clean_string_array(column) -%}
  array_distinct(
    transform(
      filter(coalesce({{ column }}, cast(array[] as array(varchar))), x -> nullif(trim(x), '') is not null),
      x -> trim(x)
    )
  )
{%- endmacro %}

{#-
  Reserved ids of every dense vocabulary (lkp_*): 0 = padding (fills fixed-length lists, never a
  value), 1 = out-of-vocabulary (a value with no id). Real ids start at 2.
-#}
{% macro padding_id() %}0{% endmacro %}
{% macro oov_id() %}1{% endmacro %}

{#- Map a string array to lookup ids; `lookup_map` is a map(varchar, bigint). Unknown -> OOV. -#}
{% macro encode_array(column, lookup_map) -%}
  transform({{ column }}, x -> coalesce(element_at({{ lookup_map }}, x), cast({{ oov_id() }} as bigint)))
{%- endmacro %}

{#- Right-pad with the padding id / truncate an array(bigint) to exactly n elements. -#}
{% macro pad_ids(column, n) -%}
  slice(coalesce({{ column }}, cast(array[] as array(bigint))) || repeat(cast({{ padding_id() }} as bigint), {{ n }}), 1, {{ n }})
{%- endmacro %}

{#- Latest value of `column` already loaded in this model, with a lower bound for empty tables. -#}
{% macro incremental_max(column, default) -%}
  (select coalesce(max({{ column }}), {{ default }}) from {{ this }})
{%- endmacro %}
