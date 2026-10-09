"""The inference -> serving interface: serving (steam_serving, a dev dependency) must read every
item this pipeline writes, and serve them end to end."""

import json

import boto3
from conftest import BUCKET, DETAILS_TABLE, N_GAMES, TABLE, ReverseLlm
from steam_serving.app import App
from steam_serving.config import ServingSettings
from steam_serving.contracts import GameDetails, StoredRecommendations
from steam_serving.repository import DynamoRepository
from steam_serving.users import UserIndex

from steam_inference.online import S3BundleStore
from steam_inference.pipeline import run_inference
from steam_inference.writer import DynamoWriter


def _run(settings, source, store):
    run_inference(
        settings,
        source,
        store,
        DynamoWriter(TABLE, region="us-east-1", concurrency=2),
        ReverseLlm(),
        details_writer=DynamoWriter(
            DETAILS_TABLE, region="us-east-1", concurrency=2, key="game_id"
        ),
        bundle_store=S3BundleStore(BUCKET, "serving/online", region="us-east-1"),
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
    users = UserIndex(BUCKET, "serving/users", region="us-east-1", refresh_seconds=300)
    repo = DynamoRepository(TABLE, DETAILS_TABLE, region="us-east-1")
    app = App(ServingSettings(), repo, users=users)

    # demo user numbers (spec 13): 104 (18 reviews), 101 (7), 102, 103, 105
    status, data = _get(app, "/users/2/recommendations")
    assert (status, data["user_id"], data["max_user"]) == (200, "101", 5)
    assert data["source"] == "personalized" and data["reranked"]
    top = data["recommendations"][0]
    assert top["explanation"] and top["details"]["name"] == top["name"]

    status, data = _get(app, "/users/5/recommendations")  # 105: no user_features, fallback
    assert (status, data["user_id"], data["source"]) == (200, "105", "popular")
    assert data["recommendations"][0]["details"]["header_image"].startswith("https://")
