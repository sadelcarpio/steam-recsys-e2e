"""Pydantic contracts passed between pipeline steps and the raw record schemas."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

RUN_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,80}$"

GAMES_PARTITION_RE = re.compile(r"^games/(?P<run_id>[^/]+)/appids-(?P<n>\d+)\.json$")
REVIEWS_PARTITION_RE = re.compile(r"^reviews/(?P<run_id>[^/]+)/part-(?P<n>\d+)\.json$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---- Step Functions contracts -------------------------------------------------------------


class ListPartitionEvent(BaseModel):
    """Lambda input. Extra keys from the state machine input are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    run_id: str = Field(pattern=RUN_ID_PATTERN)


class ListPartitionResult(_Strict):
    run_id: str
    new_game_ids: int
    games_to_scrape: int
    reviews_game_ids: int
    games_partitions: list[str]
    reviews_partitions: list[str]


class PartitionFile(_Strict):
    """Body of every `s3://game-partitions-*/…/*.json` file."""

    run_id: str
    appids: list[int]


# ---- DynamoDB state -----------------------------------------------------------------------


class GameStatus(StrEnum):
    PENDING = "pending"  # known, game info not scraped yet (or retrying)
    SCRAPED = "scraped"
    UNAVAILABLE = "unavailable"  # appdetails success=false / not a game; never retried
    FAILED = "failed"  # exhausted max_game_attempts


class GameState(_Strict):
    appid: int
    status: GameStatus
    attempts: int = 0
    recommendations: int | None = None


# ---- Raw parquet records -------------------------------------------------------------------


class GameRecord(_Strict):
    appid: int
    name: str | None
    type: str | None
    required_age: int | None
    is_free: bool | None
    minimum_pc_requirements: str | None
    recommended_pc_requirements: str | None
    controller_support: str | None
    detailed_description: str | None
    about_the_game: str | None
    short_description: str | None
    supported_languages: list[str]
    header_image: str | None
    developers: list[str]
    publishers: list[str]
    price: float | None
    categories: list[str]
    genres: list[str]
    windows_support: bool | None
    mac_support: bool | None
    linux_support: bool | None
    release_date: str | None
    coming_soon: bool | None
    recommendations: int | None
    dlc: list[int]
    review_score: int | None
    review_score_desc: str | None
    scrape_date: date


class ReviewRecord(_Strict):
    rec_id: int
    author_id: int
    appid: int
    playtime_forever: int | None
    playtime_last_two_weeks: int | None
    playtime_at_review: int | None
    num_games_owned: int | None
    num_reviews: int | None
    last_played: int | None
    language: str | None
    review: str | None
    timestamp_created: int
    timestamp_updated: int | None
    voted_up: bool | None
    votes_up: int | None
    votes_funny: int | None
    weighted_vote_score: float | None
    comment_count: int | None
    steam_purchase: bool | None
    received_for_free: bool | None
    written_during_early_access: bool | None
    primarily_steam_deck: bool | None
    scrape_date: date
