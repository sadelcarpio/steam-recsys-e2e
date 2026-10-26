-- Published Steam user tags (spec 16): the latest scrape of each published game, with tag
-- names (the mart `game_tags` only has encoded ids), by weight descending.
select
    t.appid as game_id,
    t.tag_names,
    t.tag_weights,
    cast(from_unixtime(t.scraped_at) as timestamp) as scraped_at
from (
    select
        *,
        row_number() over (partition by appid order by scraped_at desc) as scrape_rank
    from {raw}.game_tags
) as t
where t.scrape_rank = 1
    and cardinality(t.tag_names) > 0
    and t.appid in (select game_id from {marts}.game_details)
