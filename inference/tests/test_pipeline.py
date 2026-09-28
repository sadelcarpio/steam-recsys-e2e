import json
from datetime import UTC, datetime
from decimal import Decimal

import boto3
import pytest
from conftest import (
    DETAILS_TABLE,
    FIRST_GAME,
    N_GAMES,
    REVIEWS,
    TABLE,
    make_metadata,
    make_model,
    reverse_ranking,
)

from steam_inference.contracts import POPULAR_USER_ID
from steam_inference.pipeline import run_inference
from steam_inference.writer import DynamoWriter, JsonlWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)


def _writer(settings):
    return DynamoWriter(TABLE, region="us-east-1", concurrency=settings.write_concurrency)


def _items(popular: bool = False) -> dict[str, dict]:
    """Stored items by user id (the "__popular__" fallback item only when `popular`)."""
    table = boto3.resource("dynamodb").Table(TABLE)
    items = {item["user_id"]: item for item in table.scan()["Items"]}
    return items if popular else {u: i for u, i in items.items() if u != POPULAR_USER_ID}


def test_skipped_without_a_champion(settings, source, store):
    calls = []
    summary = run_inference(
        settings, source, store, _writer(settings), lambda r: calls.append(r), now=NOW
    )
    assert summary.skipped and summary.model_id is None
    assert "no model" in summary.reason
    assert _items() == {} and calls == []


def test_writes_every_user_and_reranks_the_active_ones(settings, source, store, champion):
    summary = run_inference(settings, source, store, _writer(settings), reverse_ranking, now=NOW)
    assert not summary.skipped
    assert summary.model_id == "abc123"  # the champion's own id, not "champion"
    assert summary.users == 4 and summary.written == 5  # + the popularity fallback
    items = _items()
    assert set(items) == {"101", "102", "103", "104"}  # 105: no user_features

    # >= 6 reviews: 101 (7), 102 (6), 104 (18)
    assert {u for u, item in items.items() if item["reranked"]} == {"101", "102", "104"}
    assert summary.reranked_users == 3 and summary.rerank_failures == 0

    for user, item in items.items():
        recs = item["recommendations"]
        reviewed = {1000 + g for g, _ in REVIEWS[int(user)]}
        assert not reviewed & {int(r["game_id"]) for r in recs}
        assert item["model_id"] == "abc123"
        assert item["generated_at"] == NOW.isoformat()
        assert item["content_hash"]
        assert all(isinstance(r["score"], Decimal) for r in recs)
        if item["reranked"]:
            assert item["rerank_model"] == settings.bedrock_model_id
            explained = [r for r in recs if "explanation" in r]
            assert explained == recs[: min(2, len(recs))]  # explain_top_n = 2
        else:
            assert "rerank_model" not in item
            assert all("explanation" not in r for r in recs)
            scores = [r["score"] for r in recs]
            assert scores == sorted(scores, reverse=True)
    assert len(items["101"]["recommendations"]) == 5
    assert len(items["104"]["recommendations"]) == 2

    # the fake LLM reverses the retrieval order
    reranked = items["101"]["recommendations"]
    assert [r["score"] for r in reranked] == sorted(r["score"] for r in reranked)


def test_rerank_prompt_games_carry_their_short_description(settings, source, store, champion):
    requests = []

    def recording(request):
        requests.append(request)
        return reverse_ranking(request)

    run_inference(settings, source, store, _writer(settings), recording, now=NOW)
    games = [g for r in requests for g in (*r.liked, *r.candidates)]
    # odd games have a scraped description (conftest.details_table), even ones none
    assert any(g.endswith("positive reviews | Shoot & loot 7.") for g in games)
    assert any(g.endswith("positive reviews") for g in games)


def test_rerank_is_capped_to_the_most_active_users(settings, source, store, champion):
    capped = settings.model_copy(update={"rerank_max_users": 1})
    run_inference(capped, source, store, _writer(capped), reverse_ranking, now=NOW)
    assert {u for u, item in _items().items() if item["reranked"]} == {"104"}


def test_failed_rerank_falls_back_to_retrieval_order(settings, source, store, champion):
    def failing(request):
        raise RuntimeError("bedrock down")

    summary = run_inference(settings, source, store, _writer(settings), failing, now=NOW)
    assert summary.rerank_failures == 3 and summary.reranked_users == 0
    assert summary.written == 5
    assert not any(item["reranked"] for item in _items().values())


def test_rerank_disabled(settings, source, store, champion):
    summary = run_inference(settings, source, store, _writer(settings), None, now=NOW)
    assert summary.reranked_users == 0 and summary.written == 5


LATER = datetime(2024, 7, 8, tzinfo=UTC)


def test_unchanged_users_are_not_rewritten(settings, source, store, champion):
    first = run_inference(settings, source, store, _writer(settings), None, now=NOW)
    assert (first.written, first.unchanged, first.deleted) == (5, 0, 0)
    again = run_inference(settings, source, store, _writer(settings), None, now=LATER)
    assert (again.written, again.unchanged, again.deleted) == (0, 5, 0)
    assert all(item["generated_at"] == NOW.isoformat() for item in _items(popular=True).values())


def test_changed_users_are_rewritten(settings, source, store, champion):
    run_inference(settings, source, store, _writer(settings), None, now=NOW)
    # reranking changes the order of the three active users only
    summary = run_inference(settings, source, store, _writer(settings), reverse_ranking, now=LATER)
    assert (summary.written, summary.unchanged) == (3, 2)
    items = _items()
    assert {u for u, item in items.items() if item["generated_at"] == LATER.isoformat()} == {
        "101",
        "102",
        "104",
    }
    assert items["103"]["generated_at"] == NOW.isoformat()


def test_items_without_a_hash_are_rewritten(settings, source, store, champion):
    run_inference(settings, source, store, _writer(settings), None, now=NOW)
    table = boto3.resource("dynamodb").Table(TABLE)
    table.update_item(Key={"user_id": "103"}, UpdateExpression="REMOVE content_hash")
    summary = run_inference(settings, source, store, _writer(settings), None, now=LATER)
    assert (summary.written, summary.unchanged) == (1, 4)


def test_users_that_are_gone_are_deleted(settings, source, store, champion):
    table = boto3.resource("dynamodb").Table(TABLE)
    table.put_item(Item={"user_id": "999", "content_hash": "x", "recommendations": []})
    summary = run_inference(settings, source, store, _writer(settings), None, now=NOW)
    assert summary.deleted == 1
    assert set(_items()) == {"101", "102", "103", "104"}


def test_partial_runs_never_delete(settings, source, store, champion):
    run_inference(settings, source, store, _writer(settings), None, now=NOW)
    partial = settings.model_copy(update={"max_users": 2})
    summary = run_inference(partial, source, store, _writer(partial), None, now=LATER)
    assert summary.users == 2 and summary.deleted == 0
    assert len(_items()) == 4


def test_new_champion_rewrites_everyone(settings, source, store, champion):
    run_inference(settings, source, store, _writer(settings), None, now=NOW)
    model = make_model(seed=5)
    store.save_model(model, make_metadata(model, "def456"))
    newer = settings.model_copy(update={"model_id": "def456"})
    summary = run_inference(newer, source, store, _writer(newer), None, now=LATER)
    assert (summary.written, summary.unchanged) == (4, 1)  # the fallback does not depend on it
    assert {item["model_id"] for item in _items().values()} == {"def456"}


def test_serves_a_named_model(settings, source, store):
    model = make_model(seed=3)
    store.save_model(model, make_metadata(model, "manual-1"))
    named = settings.model_copy(update={"model_id": "manual-1"})
    summary = run_inference(named, source, store, _writer(named), None, now=NOW)
    assert summary.model_id == "manual-1" and summary.written == 5


def test_rejects_an_incompatible_architecture(settings, source, store):
    model = make_model()
    metadata = make_metadata(model, "old")
    old = metadata.model_copy(
        update={"config": metadata.config.model_copy(update={"architecture_version": 0})}
    )
    store.save_model(model, old)
    with pytest.raises(RuntimeError, match="architecture"):
        run_inference(
            settings.model_copy(update={"model_id": "old"}), source, store, _writer(settings), None
        )


def test_jsonl_dry_run(settings, source, store, champion, tmp_path):
    path = tmp_path / "out" / "recs.jsonl"
    summary = run_inference(settings, source, store, JsonlWriter(str(path)), reverse_ranking)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert summary.written == len(lines) == 5
    assert isinstance(lines[0]["recommendations"][0]["score"], float)
    assert _items() == {}  # nothing written to DynamoDB


def test_popularity_fallback_item(settings, source, store, champion):
    summary = run_inference(settings, source, store, _writer(settings), reverse_ranking, now=NOW)
    assert summary.popular_games == 5
    popular = _items(popular=True)[POPULAR_USER_ID]
    assert popular["model_id"] == "popularity" and not popular["reranked"]
    recs = popular["recommendations"]
    # game 2: 3 positive reviews; then 2 each, ties by lowest appid (game 4 has 1)
    assert [int(r["game_id"]) - 1000 for r in recs] == [2, 3, 5, 6, 7]
    assert [float(r["score"]) for r in recs] == [1.0, 0.6667, 0.6667, 0.6667, 0.6667]
    assert all("explanation" not in r for r in recs)


def test_syncs_game_details_when_given_a_writer(settings, source, store, champion):
    details = DynamoWriter(DETAILS_TABLE, region="us-east-1", concurrency=2, key="game_id")
    summary = run_inference(
        settings, source, store, _writer(settings), None, details_writer=details, now=NOW
    )
    assert summary.game_details_written == N_GAMES
    assert summary.snapshots["game_details"] == 7
    again = run_inference(
        settings, source, store, _writer(settings), None, details_writer=details, now=LATER
    )
    assert again.game_details_written == 0
    stored = boto3.resource("dynamodb").Table(DETAILS_TABLE).scan()["Items"]
    assert {int(i["game_id"]) for i in stored} == {
        1000 + g for g in range(FIRST_GAME, FIRST_GAME + N_GAMES)
    }


def test_skipped_run_syncs_no_details(settings, source, store):
    details = DynamoWriter(DETAILS_TABLE, region="us-east-1", concurrency=2, key="game_id")
    summary = run_inference(
        settings, source, store, _writer(settings), None, details_writer=details, now=NOW
    )
    assert summary.skipped and summary.game_details_written == 0
