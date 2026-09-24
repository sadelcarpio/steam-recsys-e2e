"""DynamoDB reads: one GetItem per request, plus one BatchGetItem for the game details."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Protocol

import boto3
from botocore.config import Config

from steam_serving.contracts import GameDetails, StoredRecommendations

# Fail fast: a Function URL caller waits on every retry.
RETRIES = Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=2, read_timeout=3)
BATCH_ATTEMPTS = 3


class Repository(Protocol):
    def recommendations(self, user_id: str) -> StoredRecommendations | None: ...

    def game(self, game_id: int) -> GameDetails | None: ...

    def games(self, game_ids: Iterable[int]) -> dict[int, GameDetails]: ...


class DynamoRepository:
    def __init__(self, recommendations_table: str, details_table: str, *, region: str) -> None:
        self.resource = boto3.resource("dynamodb", region_name=region, config=RETRIES)
        self.recommendations_table = self.resource.Table(recommendations_table)
        self.details_table_name = details_table
        self.details_table = self.resource.Table(details_table)

    def recommendations(self, user_id: str) -> StoredRecommendations | None:
        item = self.recommendations_table.get_item(Key={"user_id": user_id}).get("Item")
        return StoredRecommendations.model_validate(item) if item else None

    def game(self, game_id: int) -> GameDetails | None:
        item = self.details_table.get_item(Key={"game_id": game_id}).get("Item")
        return GameDetails.model_validate(item) if item else None

    def games(self, game_ids: Iterable[int]) -> dict[int, GameDetails]:
        """Details of the games that have them (at most 100 ids: one BatchGetItem)."""
        keys = [{"game_id": game_id} for game_id in dict.fromkeys(game_ids)]
        found: dict[int, GameDetails] = {}
        for attempt in range(BATCH_ATTEMPTS):
            if not keys:
                break
            if attempt:
                time.sleep(0.05 * 2**attempt)
            response = self.resource.batch_get_item(
                RequestItems={self.details_table_name: {"Keys": keys}}
            )
            for item in response["Responses"].get(self.details_table_name, []):
                details = GameDetails.model_validate(item)
                found[details.game_id] = details
            keys = (
                response.get("UnprocessedKeys", {}).get(self.details_table_name, {}).get("Keys", [])
            )
        return found  # still unprocessed after the retries: served without details
