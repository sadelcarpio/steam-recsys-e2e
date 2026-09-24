-- Typed, renamed, normalized game info. Arrays: trimmed, empty strings dropped, deduplicated.
select
    cast(appid as bigint) as game_id,
    nullif(trim(name), '') as game_name,
    lower(trim(type)) as game_type,
    is_free as game_is_free,
    {{ clean_string_array('developers') }} as game_developers,
    {{ clean_string_array('publishers') }} as game_publishers,
    {{ clean_string_array('genres') }} as game_genres,
    {{ clean_string_array('categories') }} as game_categories,
    cast(review_score as bigint) as game_review_score,
    cast(recommendations as bigint) as game_recommendations,
    nullif(trim(short_description), '') as game_short_description,
    nullif(trim(header_image), '') as game_header_image,
    nullif(trim(release_date), '') as game_release_date,
    price as game_price,
    scrape_date
from {{ source('steam', 'games') }}
where appid is not null
