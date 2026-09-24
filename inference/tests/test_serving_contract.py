"""The inference -> serving interface: serving (steam_serving, a dev dependency) must read every
item this pipeline writes, and serve them end to end."""

import json

import boto3
from conftest import DETAILS_TABLE, N_GAMES, TABLE, reverse_ranking
from steam_serving.app import App
from steam_serving.config import ServingSettings
from steam_serving.contracts import GameDetails, StoredRecommendations
from steam_serving.repository import DynamoRepository

from steam_inference.pipeline import run_inference
from steam_inference.writer import DynamoWriter


def _run(settings, source, store):
    run_inference(
        settings,
        source,
        store,
        DynamoWriter(TABLE, region="us-east-1", concurrency=2),
        reverse_ranking,
        details_writer=DynamoWriter(
            DETAILS_TABLE, region="us-east-1", concurrency=2, key="game_id"
        ),
    )


def _get(app: App, path: str, query: dict | None = None) -> tuple[int, dict]:
    event = {
        "rawPath": path,
        "queryStringParameters": query,
        "requestContext": {"http": {"method": "GET"}},
    }
    response = app.handle(event)
    return response["statusCode"], json.loads(response["body"])


def test_serving_parses_every_written_item(settings, source, store, champion):
    _run(settings, source, store)
    dynamodb = boto3.resource("dynamodb")
    recs = dynamodb.Table(TABLE).scan()["Items"]
    assert len(recs) == 5
    for item in recs:
        StoredRecommendations.model_validate(item)
    details = dynamodb.Table(DETAILS_TABLE).scan()["Items"]
    assert len(details) == N_GAMES
    for item in details:
        GameDetails.model_validate(item)


def test_serves_what_inference_wrote(settings, source, store, champion, monkeypatch):
    _run(settings, source, store)
    monkeypatch.delenv("USE_SSM", raising=False)
    app = App(ServingSettings(), DynamoRepository(TABLE, DETAILS_TABLE, region="us-east-1"))

    status, data = _get(app, "/users/101/recommendations")
    assert status == 200 and data["source"] == "personalized" and data["reranked"]
    top = data["recommendations"][0]
    assert top["explanation"] and top["details"]["name"] == top["name"]

    status, data = _get(app, "/users/105/recommendations")  # no user_features: fallback
    assert status == 200 and data["source"] == "popular"
    assert data["recommendations"][0]["details"]["header_image"].startswith("https://")
