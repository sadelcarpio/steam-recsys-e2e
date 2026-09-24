import base64
import json

import pytest
from conftest import body, event

from steam_serving.app import App


def post(app, payload, *, raw: str | None = None, b64: bool = False, path="/recommendations"):
    e = event(path, method="POST")
    text = raw if raw is not None else json.dumps(payload)
    e["body"] = base64.b64encode(text.encode()).decode() if b64 else text
    e["isBase64Encoded"] = b64
    return app.handle(e)


def test_online_recommendations(online_app):
    response = post(online_app, {"liked_game_ids": [12, 11, 99], "limit": 4})
    assert response["statusCode"] == 200
    assert response["headers"]["Cache-Control"] == "no-store"
    data = body(response)
    assert (data["source"], data["model_id"], data["reranked"]) == ("online", "abc123", False)
    assert "user_id" not in data
    assert (data["used_game_ids"], data["ignored_game_ids"]) == ([12, 11], [99])
    recs = data["recommendations"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4]
    assert not {12, 11} & {r["game_id"] for r in recs}
    assert all(r["name"] == f"Game {r['game_id']} é" for r in recs)
    assert all(r["details"]["header_image"].endswith(f"{r['game_id']}.jpg") for r in recs)
    assert [r["score"] for r in recs] == sorted((r["score"] for r in recs), reverse=True)


def test_online_without_details_needs_no_dynamodb(settings, online_model):
    class NoTables:
        def games(self, ids):
            raise AssertionError("no DynamoDB call expected")

    app = App(settings, NoTables(), lambda: online_model)
    data = body(post(app, {"liked_game_ids": [12], "details": False}))
    assert len(data["recommendations"]) == 10
    assert all("details" not in r for r in data["recommendations"])


def test_base64_body(online_app):
    assert post(online_app, {"liked_game_ids": [12]}, b64=True)["statusCode"] == 200


def test_unknown_games_only_fall_back_to_popular(online_app):
    data = body(post(online_app, {"liked_game_ids": [100, 5], "limit": 3}))
    assert data["source"] == "popular" and data["model_id"] == "popularity"
    assert [r["game_id"] for r in data["recommendations"]] == [101, 102, 103]  # 100 was liked
    assert (data["used_game_ids"], data["ignored_game_ids"]) == ([], [100, 5])


def test_no_bundle_is_503(tables, settings):
    from steam_serving.repository import DynamoRepository

    repo = DynamoRepository("game-explainable-recommendations", "game-details", region="us-east-1")
    for online in (None, lambda: None):
        response = post(App(settings, repo, online), {"liked_game_ids": [12]})
        assert response["statusCode"] == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"liked_game_ids": []},
        {"liked_game_ids": ["12"]},
        {"liked_game_ids": [12.5]},
        {"liked_game_ids": [12], "limit": 0},
        {"liked_game_ids": [12], "limit": 31},
        {"liked_game_ids": [12], "details": "yes"},
        {"liked_game_ids": [12], "user_id": "1"},
        {"liked_game_ids": list(range(101))},
        [12],
    ],
)
def test_invalid_bodies(online_app, payload):
    response = post(online_app, payload)
    assert response["statusCode"] == 400
    assert body(response)["error"] == "invalid body" or body(response)["error"] == "invalid limit"


@pytest.mark.parametrize("raw", ["", "not json", "{"])
def test_unparseable_bodies(online_app, raw):
    response = post(online_app, None, raw=raw)
    assert (response["statusCode"], body(response)["error"]) == (400, "invalid body")


def test_post_elsewhere_is_405(online_app):
    assert post(online_app, {"liked_game_ids": [12]}, path="/popular")["statusCode"] == 405
