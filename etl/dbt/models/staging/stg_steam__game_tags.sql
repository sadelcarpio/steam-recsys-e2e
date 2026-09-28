-- One row per tags scrape of a game (spec 8): tag names and Steam's weights as aligned arrays,
-- in the scraper's order (weight descending). Names trimmed; pairs without a name dropped.
with scrapes as (
    select
        cast(appid as bigint) as game_id,
        filter(
            zip_with(
                coalesce(tag_names, cast(array[] as array(varchar))),
                coalesce(tag_weights, cast(array[] as array(bigint))),
                (n, w) -> cast(
                    row(nullif(trim(n), ''), coalesce(w, 0)) as row(name varchar, weight bigint)
                )
            ),
            t -> t.name is not null
        ) as tags,
        cast(scraped_at as bigint) as scraped_at,
        scrape_date
    from {{ source('steam', 'game_tags') }}
    where appid is not null
)

select
    game_id,
    transform(tags, t -> t.name) as game_tags,
    transform(tags, t -> t.weight) as game_tag_weights,
    scraped_at,
    scrape_date
from scrapes
