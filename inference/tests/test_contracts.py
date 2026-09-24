from datetime import UTC, datetime

from steam_inference.contracts import Recommendation, UserRecommendations


def _recs(**changes) -> UserRecommendations:
    base = {
        "user_id": "1",
        "recommendations": [
            Recommendation(game_id=10, name="A", score=0.5, explanation="why"),
            Recommendation(game_id=11, name="B", score=0.4),
        ],
        "model_id": "m1",
        "generated_at": datetime(2024, 1, 1, tzinfo=UTC),
        "reranked": True,
        "rerank_model": "llm",
    }
    return UserRecommendations(**{**base, **changes})


def test_hash_ignores_scores_and_generation_time():
    moved = [
        Recommendation(game_id=10, name="A", score=0.51, explanation="why"),
        Recommendation(game_id=11, name="B", score=0.39),
    ]
    same = _recs(recommendations=moved, generated_at=datetime(2024, 2, 1, tzinfo=UTC))
    assert same.content_hash() == _recs().content_hash()


def test_hash_tracks_what_the_user_sees():
    base = _recs().content_hash()
    swapped = [
        Recommendation(game_id=11, name="B", score=0.4),
        Recommendation(game_id=10, name="A", score=0.5, explanation="why"),
    ]
    variants = [
        _recs(recommendations=swapped),
        _recs(recommendations=[Recommendation(game_id=10, name="A", score=0.5, explanation="x")]),
        _recs(model_id="m2"),
        _recs(reranked=False, rerank_model=None),
    ]
    assert all(v.content_hash() != base for v in variants)


def test_item_carries_the_hash():
    item = _recs().to_item()
    assert item["content_hash"] == _recs().content_hash()
    assert "explanation" not in item["recommendations"][1]
