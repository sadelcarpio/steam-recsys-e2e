-- Published reviews (spec 16): raw reviews deduplicated with the rule of
-- int_reviews__deduplicated (one row per review_id, the first scrape wins, then the latest
-- update) and the filters of stg_steam__reviews. `user_id` is the scraper's keyed hash
-- (spec 13). Never add scrape bookkeeping, `last_played` or any demo user number.
select
    review_id,
    user_id,
    game_id,
    is_positive,
    review,
    language,
    reviewed_at,
    updated_at,
    playtime_forever,
    playtime_last_two_weeks,
    playtime_at_review,
    num_games_owned,
    num_reviews,
    votes_up,
    votes_funny,
    weighted_vote_score,
    comment_count,
    steam_purchase,
    received_for_free,
    written_during_early_access,
    primarily_steam_deck
from (
    select
        *,
        row_number() over (
            partition by review_id order by scrape_date asc, updated_at desc
        ) as seen_rank
    from (
        select
            cast(rec_id as bigint) as review_id,
            cast(user_id as bigint) as user_id,
            cast(appid as bigint) as game_id,
            voted_up as is_positive,
            review,
            language,
            cast(from_unixtime(timestamp_created) as timestamp) as reviewed_at,
            cast(from_unixtime(coalesce(timestamp_updated, timestamp_created)) as timestamp)
                as updated_at,
            playtime_forever,
            playtime_last_two_weeks,
            playtime_at_review,
            num_games_owned,
            num_reviews,
            votes_up,
            votes_funny,
            weighted_vote_score,
            comment_count,
            steam_purchase,
            received_for_free,
            written_during_early_access,
            primarily_steam_deck,
            scrape_date
        from {raw}.reviews
        where rec_id is not null
            and user_id is not null
            and appid is not null
            and voted_up is not null
            and timestamp_created > 0
    )
)
where seen_rank = 1
