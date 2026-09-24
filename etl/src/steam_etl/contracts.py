"""Row contracts of the Iceberg marts (`<schema>_marts.*`), the ETL -> training/inference interface.

Ids from the lookup tables (lkp_*) are dense and start at 2. Two ids are reserved and mean
different things: PADDING_ID fills fixed-length lists and is never a value, OOV_ID stands for a
value that has no id (e.g. a vocabulary entry newer than a trained model).
Keep in sync with dbt/models/marts/*.sql (checked by the Athena integration test).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

PADDING_ID = 0
OOV_ID = 1
FIRST_ID = 2
# dbt var `user_history_length`.
USER_HISTORY_LENGTH = 5

Id = Annotated[int, Field(ge=FIRST_ID)]
# Encoded attribute values: a real id, or OOV when the name has no id. Never padding.
EncodedId = Annotated[int, Field(ge=OOV_ID)]


def _padding_or_id(value: int) -> int:
    if value != PADDING_ID and value < FIRST_ID:
        raise ValueError(f"history entries are {PADDING_ID} (padding) or >= {FIRST_ID}")
    return value


# Games in a history always come from lkp_games, so they are never OOV.
HistoryEntry = Annotated[int, AfterValidator(_padding_or_id)]
History = Annotated[
    list[HistoryEntry], Field(min_length=USER_HISTORY_LENGTH, max_length=USER_HISTORY_LENGTH)
]


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_at: datetime = Field(alias="_batch_at")


class LookupRow(_Row):
    """lkp_developers / lkp_publishers / lkp_genres / lkp_categories."""

    id: Id
    name: str


class GameLookupRow(_Row):
    """lkp_games."""

    game_idx: Id
    game_id: int


class _GameFeatures(BaseModel):
    game_idx: Id
    game_name: str
    game_is_free: bool | None
    game_developers: list[EncodedId]
    game_publishers: list[EncodedId]
    game_genres: list[EncodedId]
    game_categories: list[EncodedId]
    # Laplace-smoothed (pos + a) / (pos + neg + 2a); 0.5 before the first review.
    game_reviews_ratio: Annotated[float, Field(gt=0, lt=1)]


class GameFeaturesRow(_GameFeatures, _Row):
    """game_features: state of a game through `timestamp` (latest row = current state)."""

    game_id: int
    timestamp: datetime


class UserFeaturesRow(_Row):
    """user_features: state of a user through `timestamp`."""

    user_id: int
    timestamp: datetime
    # Most recent first, right-padded with PADDING_ID.
    games_reviewed_positive: History


class InteractionRow(_GameFeatures, _Row):
    """interactions: one review with user/game features as of strictly before it."""

    review_id: int
    timestamp: datetime
    user_id: int
    game_id: int
    is_positive: bool
    games_reviewed_positive: History


MART_CONTRACTS: dict[str, type[_Row]] = {
    "lkp_developers": LookupRow,
    "lkp_publishers": LookupRow,
    "lkp_genres": LookupRow,
    "lkp_categories": LookupRow,
    "lkp_games": GameLookupRow,
    "game_features": GameFeaturesRow,
    "user_features": UserFeaturesRow,
    "interactions": InteractionRow,
}
