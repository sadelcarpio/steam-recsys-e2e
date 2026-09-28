import pytest
from conftest import GENERATED_AT, body, event

from steam_serving.contracts import GameDetails, RecommendationsResponse

USER = "76561198000000001"


def test_health_needs_no_tables(settings):
    from steam_serving.app import App

    response = App(settings, repository=None).handle(event("/health"))
    assert response["statusCode"] == 200 and body(response) == {"status": "ok"}
    assert response["headers"]["Cache-Control"] == "no-store"


def test_user_recommendations_with_details(app):
    response = app.handle(event(f"/users/{USER}/recommendations"))
    assert response["statusCode"] == 200
    assert response["headers"]["Content-Type"] == "application/json"
    assert response["headers"]["Cache-Control"] == "public, max-age=300"
    data = RecommendationsResponse.model_validate_json(response["body"])
    assert (data.source, data.user_id, data.model_id) == ("personalized", USER, "abc123")
    assert data.reranked and data.rerank_model == "us.amazon.nova-2-lite-v1:0"
    assert data.generated_at.isoformat() == GENERATED_AT
    recs = data.recommendations
    assert len(recs) == 10  # default limit
    assert [r.rank for r in recs] == list(range(1, 11))
    assert [r.game_id for r in recs] == list(range(10, 20))
    assert [r.explanation for r in recs[:3]] == ["why 10", "why 11", None]
    assert recs[0].score == pytest.approx(0.9)
    assert recs[0].details.header_image == "https://cdn/10.jpg"
    assert recs[0].details.price == pytest.approx(9.99)
    assert recs[1].details.short_description == "About 11."
    raw = body(response)["recommendations"]
    assert "explanation" not in raw[2]  # nulls are omitted
    assert "short_description" not in raw[0]["details"]


def test_limit_and_no_details(app):
    data = body(
        app.handle(event(f"/users/{USER}/recommendations", {"limit": "30", "details": "false"}))
    )
    assert len(data["recommendations"]) == 30
    assert all("details" not in r for r in data["recommendations"])


def test_short_list_is_returned_whole(app):
    data = body(app.handle(event("/users/76561198000000002/recommendations", {"limit": "20"})))
    assert [r["game_id"] for r in data["recommendations"]] == [10, 11, 12]
    assert data["reranked"] is False and "rerank_model" not in data


def test_unknown_user_gets_the_popularity_fallback(app):
    data = body(app.handle(event("/users/123/recommendations", {"limit": "7"})))
    assert (data["source"], data["user_id"], data["model_id"]) == ("popular", "123", "popularity")
    recs = data["recommendations"]
    assert [r["game_id"] for r in recs] == list(range(100, 107))
    assert "details" in recs[4] and "details" not in recs[5]  # 105+ have no details item


def test_popular(app):
    data = body(app.handle(event("/popular")))
    assert data["source"] == "popular" and "user_id" not in data
    assert len(data["recommendations"]) == 10


def test_nothing_written_yet_is_404(app, tables):
    recs, _ = tables
    recs.delete_item(Key={"user_id": "__popular__"})
    for path in ("/popular", "/users/123/recommendations"):
        response = app.handle(event(path))
        assert response["statusCode"] == 404
        assert body(response)["error"] == "no recommendations"
    # a known user is still served
    assert app.handle(event(f"/users/{USER}/recommendations"))["statusCode"] == 200


def test_popular_user_id_is_not_addressable(app):
    response = app.handle(event("/users/__popular__/recommendations"))
    assert response["statusCode"] == 400


def test_game(app):
    response = app.handle(event("/games/11"))
    game = GameDetails.model_validate_json(response["body"])
    assert game.name == "Game 11" and game.genres == ["Action"]
    assert app.handle(event("/games/999"))["statusCode"] == 404
    assert app.handle(event("/games/abc"))["statusCode"] == 400


@pytest.mark.parametrize(
    ("path", "query", "status"),
    [
        (f"/users/{USER}/recommendations", {"limit": "0"}, 400),
        (f"/users/{USER}/recommendations", {"limit": "31"}, 400),
        (f"/users/{USER}/recommendations", {"limit": "x"}, 400),
        # str.isdigit() accepts these, int() does not (was a 500)
        (f"/users/{USER}/recommendations", {"limit": "²"}, 400),
        ("/popular", {"limit": "٣"}, 400),
        ("/games/²", None, 400),
        (f"/users/{USER}/recommendations", {"details": "maybe"}, 400),
        ("/users/12a/recommendations", None, 400),
        ("/users/123456789012345678901/recommendations", None, 400),
        ("/nope", None, 404),
        ("/", None, 404),
    ],
)
def test_bad_requests(app, path, query, status):
    response = app.handle(event(path, query))
    assert response["statusCode"] == status
    assert body(response)["error"]
    assert response["headers"]["Cache-Control"] == "no-store"


def test_only_get(app):
    response = app.handle(event(f"/users/{USER}/recommendations", method="POST"))
    assert response["statusCode"] == 405


def test_internal_errors_are_500_without_details(settings):
    from steam_serving.app import App

    class Broken:
        def recommendations(self, user_id):
            raise RuntimeError("secret internals")

    response = App(settings, Broken()).handle(event("/popular"))
    assert response["statusCode"] == 500
    assert body(response) == {"error": "internal error"}
