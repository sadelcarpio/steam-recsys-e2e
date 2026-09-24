from datetime import datetime

import pytest
from pydantic import ValidationError

from steam_etl.contracts import OOV_ID, PADDING_ID, InteractionRow, LookupRow

NOW = datetime(2026, 1, 1)


def interaction(**overrides):
    row = {
        "_batch_at": NOW,
        "review_id": 1,
        "timestamp": NOW,
        "user_id": 100,
        "game_id": 10,
        "game_idx": 2,
        "is_positive": True,
        "games_reviewed_positive": [3, 2, 0, 0, 0],
        "game_name": "Alpha",
        "game_is_free": False,
        "game_developers": [2, OOV_ID],
        "game_publishers": [],
        "game_genres": [4],
        "game_categories": [],
        "game_reviews_ratio": 0.5,
    }
    return InteractionRow.model_validate(row | overrides)


def test_valid_row():
    assert interaction().games_reviewed_positive == [3, 2, 0, 0, 0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"games_reviewed_positive": [3, OOV_ID, 0, 0, 0]},  # OOV is not padding
        {"games_reviewed_positive": [3, 2, 0, 0]},  # fixed length
        {"game_developers": [PADDING_ID]},  # padding is never a value
        {"game_idx": OOV_ID},
        {"game_reviews_ratio": 1.0},  # smoothed: never 0 or 1
        {"game_reviews_ratio": None},
        {"game_positive_reviews": 3},  # raw counts are not a feature
        {"game_review_score": 8},  # scraped once: would leak future reviews
        {"unexpected": 1},
    ],
)
def test_rejects(overrides):
    with pytest.raises(ValidationError):
        interaction(**overrides)


def test_lookup_ids_start_after_reserved():
    with pytest.raises(ValidationError):
        LookupRow.model_validate({"_batch_at": NOW, "id": OOV_ID, "name": "x"})
    assert LookupRow.model_validate({"_batch_at": NOW, "id": 2, "name": "x"}).id == 2
