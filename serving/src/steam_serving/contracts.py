"""Pydantic contracts of the serving API.

Stored items (read): the tables written by the inference pipeline. Their source of truth is
inference/src/steam_inference/contracts.py; inference's `tests/test_serving_contract.py` checks
that these models parse every item it writes. Readers are tolerant (unknown fields ignored), so
inference can add fields first and serving use them later.

Responses (written): the JSON returned by the Function URL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

POPULAR_USER_ID = "__popular__"
# Steam 64-bit account ids (a string: exceeds JavaScript's safe integers).
USER_ID_PATTERN = r"^[0-9]{1,20}$"


class _Stored(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class StoredRecommendation(_Stored):
    game_id: int
    name: str
    score: float
    explanation: str | None = None


class StoredRecommendations(_Stored):
    """Table `game-explainable-recommendations`, one item per user (+ `__popular__`)."""

    user_id: str
    recommendations: list[StoredRecommendation]
    model_id: str
    generated_at: datetime
    reranked: bool
    rerank_model: str | None = None


class GameDetails(_Stored):
    """Table `game-details`, one item per catalog game (also the `/games/{id}` response)."""

    game_id: int
    name: str
    short_description: str | None = None
    header_image: str | None = None
    release_date: str | None = None
    is_free: bool | None = None
    price: float | None = None
    developers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecommendationOut(_Response):
    rank: int  # 1-based
    game_id: int
    name: str
    # Retrieval: cosine similarity of the user and the game. Popular: share of the top game's
    # recent positive reviews.
    score: float
    explanation: str | None = None  # LLM explanation (reranked users, top N only)
    details: GameDetails | None = None  # when requested and the game has details


class OnlineRequest(BaseModel):
    """Body of POST /recommendations: recommendations for a history given in the request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Steam appids the user liked, most recent first (the order feeds the model's history).
    liked_game_ids: list[StrictInt] = Field(min_length=1)
    limit: StrictInt | None = None
    details: StrictBool | None = None


class RecommendationsResponse(_Response):
    # "personalized": the user's own list. "popular": the fallback for users without one (and
    # the `/popular` endpoint, where user_id is null). "online": computed for the request's
    # liked games (POST /recommendations).
    source: Literal["personalized", "popular", "online"]
    user_id: str | None
    model_id: str
    generated_at: datetime
    reranked: bool
    rerank_model: str | None = None
    recommendations: list[RecommendationOut]
    # POST /recommendations only: liked games that made up the history / that were unknown
    used_game_ids: list[int] | None = None
    ignored_game_ids: list[int] | None = None


class ErrorResponse(_Response):
    error: str
    detail: str | None = None
