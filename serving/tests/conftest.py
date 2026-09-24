"""moto DynamoDB tables seeded with items shaped like the inference pipeline's output."""

from __future__ import annotations

import json
from decimal import Decimal

import boto3
import pytest

from steam_serving.app import App
from steam_serving.config import ServingSettings
from steam_serving.repository import DynamoRepository

RECS_TABLE = "game-explainable-recommendations"
DETAILS_TABLE = "game-details"
GENERATED_AT = "2026-09-24T17:00:00+00:00"


def user_item(user_id: str, games: list[int], *, reranked: bool) -> dict:
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "game_id": g,
                "name": f"Game {g}",
                "score": Decimal("0.9") - Decimal(i) / 100,
                **({"explanation": f"why {g}"} if reranked and i < 2 else {}),
            }
            for i, g in enumerate(games)
        ],
        "model_id": "abc123",
        "generated_at": GENERATED_AT,
        "reranked": reranked,
        **({"rerank_model": "us.amazon.nova-2-lite-v1:0"} if reranked else {}),
        "content_hash": "0" * 32,
    }


def popular_item(games: list[int]) -> dict:
    item = user_item("__popular__", games, reranked=False)
    return {**item, "model_id": "popularity"}


def details_item(game_id: int) -> dict:
    item = {
        "game_id": game_id,
        "name": f"Game {game_id}",
        "header_image": f"https://cdn/{game_id}.jpg",
        "is_free": False,
        "price": Decimal("9.99"),
        "developers": ["Studio"],
        "publishers": [],
        "genres": ["Action"],
        "categories": [],
    }
    if game_id % 2:
        item["short_description"] = f"About {game_id}."
    return item


def event(path: str, query: dict | None = None, method: str = "GET") -> dict:
    """Function URL event, payload format 2.0."""
    return {
        "version": "2.0",
        "rawPath": path,
        "queryStringParameters": query,
        "requestContext": {"http": {"method": method, "path": path}},
    }


def body(response: dict) -> dict:
    return json.loads(response["body"])


@pytest.fixture
def aws(monkeypatch):
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("USE_SSM", raising=False)
    with mock_aws():
        client = boto3.client("dynamodb")
        for name, key, key_type in ((RECS_TABLE, "user_id", "S"), (DETAILS_TABLE, "game_id", "N")):
            client.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": key_type}],
                BillingMode="PAY_PER_REQUEST",
            )
        yield


@pytest.fixture
def tables(aws):
    dynamodb = boto3.resource("dynamodb")
    recs, details = dynamodb.Table(RECS_TABLE), dynamodb.Table(DETAILS_TABLE)
    recs.put_item(Item=user_item("76561198000000001", list(range(10, 40)), reranked=True))
    recs.put_item(Item=user_item("76561198000000002", [10, 11, 12], reranked=False))
    recs.put_item(Item=popular_item(list(range(100, 130))))
    for game_id in [*range(10, 40), *range(100, 105)]:  # no details for 105..129
        details.put_item(Item=details_item(game_id))
    return recs, details


@pytest.fixture
def settings() -> ServingSettings:
    return ServingSettings()


@pytest.fixture
def app(tables, settings) -> App:
    return App(settings, DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1"))
