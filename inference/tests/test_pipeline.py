import json
from datetime import UTC, datetime
from decimal import Decimal

import boto3
import pytest
from conftest import REVIEWS, TABLE, make_metadata, make_model, reverse_ranking

from steam_inference.pipeline import run_inference
from steam_inference.writer import DynamoWriter, JsonlWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)


def _writer(settings):
    return DynamoWriter(TABLE, region="us-east-1", concurrency=settings.write_concurrency)


def _items() -> dict[str, dict]:
    table = boto3.resource("dynamodb").Table(TABLE)
    return {item["user_id"]: item for item in table.scan()["Items"]}


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
    assert summary.users == 4 and summary.written == 4
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
        assert item["expires_at"] == int(NOW.timestamp()) + 21 * 86400
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


def test_rerank_is_capped_to_the_most_active_users(settings, source, store, champion):
    capped = settings.model_copy(update={"rerank_max_users": 1})
    run_inference(capped, source, store, _writer(capped), reverse_ranking, now=NOW)
    assert {u for u, item in _items().items() if item["reranked"]} == {"104"}


def test_failed_rerank_falls_back_to_retrieval_order(settings, source, store, champion):
    def failing(request):
        raise RuntimeError("bedrock down")

    summary = run_inference(settings, source, store, _writer(settings), failing, now=NOW)
    assert summary.rerank_failures == 3 and summary.reranked_users == 0
    assert summary.written == 4
    assert not any(item["reranked"] for item in _items().values())


def test_rerank_disabled(settings, source, store, champion):
    summary = run_inference(settings, source, store, _writer(settings), None, now=NOW)
    assert summary.reranked_users == 0 and summary.written == 4


def test_rerun_overwrites(settings, source, store, champion):
    run_inference(settings, source, store, _writer(settings), reverse_ranking, now=NOW)
    later = datetime(2024, 7, 8, tzinfo=UTC)
    run_inference(settings, source, store, _writer(settings), None, now=later)
    items = _items()
    assert len(items) == 4
    assert all(item["generated_at"] == later.isoformat() for item in items.values())
    assert not any(item["reranked"] for item in items.values())


def test_serves_a_named_model(settings, source, store):
    model = make_model(seed=3)
    store.save_model(model, make_metadata(model, "manual-1"))
    named = settings.model_copy(update={"model_id": "manual-1"})
    summary = run_inference(named, source, store, _writer(named), None, now=NOW)
    assert summary.model_id == "manual-1" and summary.written == 4


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
    no_ttl = settings.model_copy(update={"ttl_days": 0})
    summary = run_inference(no_ttl, source, store, JsonlWriter(str(path)), reverse_ranking)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert summary.written == len(lines) == 4
    assert all("expires_at" not in line for line in lines)
    assert isinstance(lines[0]["recommendations"][0]["score"], float)
    assert _items() == {}  # nothing written to DynamoDB
