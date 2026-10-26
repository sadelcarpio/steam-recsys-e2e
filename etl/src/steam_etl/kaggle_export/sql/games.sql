-- Published games (spec 16): the display mart, one row per game (`game_coming_soon` = not
-- released at its latest scrape). `game_name_key` is an internal merge key, left out.
select
    game_id,
    game_name,
    game_short_description,
    game_header_image,
    game_release_date,
    game_is_free,
    game_price,
    game_developers,
    game_publishers,
    game_genres,
    game_categories,
    game_coming_soon
from {marts}.game_details
