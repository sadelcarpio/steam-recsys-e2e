"""Polars schemas of the raw parquet files (the contract consumed by the ETL)."""

import polars as pl

GAMES_SCHEMA = pl.Schema(
    {
        "appid": pl.Int64,
        "name": pl.Utf8,
        "type": pl.String,
        "required_age": pl.Int64,
        "is_free": pl.Boolean,
        "minimum_pc_requirements": pl.Utf8,
        "recommended_pc_requirements": pl.Utf8,
        "controller_support": pl.String,
        "detailed_description": pl.Utf8,
        "about_the_game": pl.Utf8,
        "short_description": pl.Utf8,
        "supported_languages": pl.List(pl.String),
        "header_image": pl.String,
        "developers": pl.List(pl.String),
        "publishers": pl.List(pl.String),
        "price": pl.Float64,
        "categories": pl.List(pl.String),
        "genres": pl.List(pl.String),
        "windows_support": pl.Boolean,
        "mac_support": pl.Boolean,
        "linux_support": pl.Boolean,
        "release_date": pl.String,
        "coming_soon": pl.Boolean,
        "recommendations": pl.Int64,
        "dlc": pl.List(pl.Int64),
        "review_score": pl.Int64,
        "review_score_desc": pl.String,
        "scrape_date": pl.Date,
    }
)

REVIEWS_SCHEMA = pl.Schema(
    {
        "rec_id": pl.Int64,
        "author_id": pl.Int64,
        "appid": pl.Int64,
        "playtime_forever": pl.Int64,
        "playtime_last_two_weeks": pl.Int64,
        "playtime_at_review": pl.Int64,
        "num_games_owned": pl.Int64,
        "num_reviews": pl.Int64,
        "last_played": pl.Int64,
        "language": pl.String,
        "review": pl.Utf8,
        "timestamp_created": pl.Int64,
        "timestamp_updated": pl.Int64,
        "voted_up": pl.Boolean,
        "votes_up": pl.Int64,
        "votes_funny": pl.Int64,
        "weighted_vote_score": pl.Float64,
        "comment_count": pl.Int64,
        "steam_purchase": pl.Boolean,
        "received_for_free": pl.Boolean,
        "written_during_early_access": pl.Boolean,
        "primarily_steam_deck": pl.Boolean,
        "scrape_date": pl.Date,
    }
)
