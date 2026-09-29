from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import responses
from responses import matchers

from steam_ingestion.config import IngestionSettings
from steam_ingestion.list_partition_game_ids import handler as lambda_handler
from steam_ingestion.list_partition_game_ids.handler import run
from steam_ingestion.state import CATALOG_CURSOR_APPID
from steam_ingestion.steam_api import APP_LIST_URL, SteamClient
from steam_ingestion.storage import list_keys, read_partition

from .conftest import PARTITIONS_BUCKET


def _mock_app_list(apps: list[tuple[int, int]], modified_since: int | None = None) -> None:
    params = {"if_modified_since": str(modified_since)} if modified_since else {}
    responses.get(
        APP_LIST_URL,
        match=[matchers.query_param_matcher(params, strict_match=False)],
        json={"response": {"apps": [{"appid": a, "last_modified": m} for a, m in apps]}},
    )


def _partitions(aws: SimpleNamespace, prefix: str) -> dict[str, list[int]]:
    return {
        k: read_partition(aws.s3, PARTITIONS_BUCKET, k).appids
        for k in list_keys(aws.s3, PARTITIONS_BUCKET, prefix)
    }


@responses.activate
def test_first_run_partitions_whole_catalog(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _mock_app_list([(i, 100 + i) for i in range(1, 11)])
    result = run("run-1", settings, client, aws.s3, aws.dynamodb)

    assert result.new_game_ids == 10 and result.games_to_scrape == 10
    games = _partitions(aws, "games/run-1/")
    assert list(games) == [f"games/run-1/appids-{n:03d}.json" for n in range(3)]  # ≤4 per task
    assert sorted(sum(games.values(), [])) == list(range(1, 11))
    reviews = _partitions(aws, "reviews/run-1/")
    assert len(reviews) == 3
    assert sorted(sum(reviews.values(), [])) == list(range(1, 11))
    assert result.reviews_partitions == sorted(reviews)

    cursor = aws.games.get_item(Key={"appid": CATALOG_CURSOR_APPID})["Item"]
    assert int(cursor["last_modified"]) == 110
    assert aws.games.get_item(Key={"appid": 5})["Item"]["status"] == "pending"


@responses.activate
def test_catalog_cursor_item_not_returned_as_game(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _mock_app_list([(1, 100)])
    run("run-1", settings, client, aws.s3, aws.dynamodb)
    responses.reset()
    _mock_app_list([], modified_since=100)
    result = run("run-2", settings, client, aws.s3, aws.dynamodb)
    assert sum(_partitions(aws, "reviews/run-2/").values(), []) == [1]
    assert result.reviews_game_ids == 1


@responses.activate
def test_incremental_run_only_scrapes_new_and_pending(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    for appid, status in [(1, "scraped"), (2, "pending"), (3, "unavailable"), (4, "failed")]:
        aws.games.put_item(Item={"appid": appid, "status": status, "attempts": 0})
    aws.games.put_item(Item={"appid": CATALOG_CURSOR_APPID, "last_modified": 500})
    # appid 1 was modified (known), appid 9 is new.
    _mock_app_list([(1, 600), (9, 700)], modified_since=500)

    result = run("run-2", settings, client, aws.s3, aws.dynamodb)

    assert result.new_game_ids == 1
    assert sum(_partitions(aws, "games/run-2/").values(), []) == [2, 9]
    # Unavailable games are not review-scraped; failed game-info scrapes still are.
    assert sorted(sum(_partitions(aws, "reviews/run-2/").values(), [])) == [1, 2, 4, 9]
    cursor = aws.games.get_item(Key={"appid": CATALOG_CURSOR_APPID})["Item"]
    assert int(cursor["last_modified"]) == 700


@responses.activate
def test_changed_coming_soon_games_are_rescraped_and_never_review_scraped(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    states = [
        (1, "scraped", True),  # coming soon, changed -> re-scraped
        (2, "scraped", True),  # coming soon, unchanged -> left alone
        (3, "scraped", False),  # released, changed -> no re-scrape (prices are out of scope)
        (4, "failed", True),  # not scraped -> not re-queued
    ]
    for appid, status, coming_soon in states:
        aws.games.put_item(
            Item={"appid": appid, "status": status, "attempts": 3, "coming_soon": coming_soon}
        )
    aws.games.put_item(Item={"appid": CATALOG_CURSOR_APPID, "last_modified": 500})
    _mock_app_list([(1, 600), (3, 610), (4, 620), (9, 700)], modified_since=500)

    result = run("run-8", settings, client, aws.s3, aws.dynamodb)

    assert result.rescrape_game_ids == 1 and result.games_to_scrape == 2
    assert sum(_partitions(aws, "games/run-8/").values(), []) == [1, 9]
    item = aws.games.get_item(Key={"appid": 1})["Item"]
    assert item["status"] == "pending" and int(item["attempts"]) == 0
    assert aws.games.get_item(Key={"appid": 4})["Item"]["status"] == "failed"
    # unreleased games have no reviews to scrape (the new game 9 is not known to be unreleased)
    assert sorted(sum(_partitions(aws, "reviews/run-8/").values(), [])) == [3, 9]


@responses.activate
def test_no_new_games_writes_no_game_partitions(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    aws.games.put_item(Item={"appid": 1, "status": "scraped", "attempts": 0})
    _mock_app_list([])
    result = run("run-3", settings, client, aws.s3, aws.dynamodb)
    assert result.games_partitions == []
    assert list_keys(aws.s3, PARTITIONS_BUCKET, "games/") == []
    assert len(result.reviews_partitions) == 1


@responses.activate
def test_rerun_same_run_id_is_idempotent(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _mock_app_list([(i, 100 + i) for i in range(1, 11)])
    first = run("run-1", settings, client, aws.s3, aws.dynamodb)
    first_files = {**_partitions(aws, "games/run-1/"), **_partitions(aws, "reviews/run-1/")}
    # Retry after the cursor advanced: GetAppList now returns nothing new.
    responses.reset()
    _mock_app_list([], modified_since=110)
    second = run("run-1", settings, client, aws.s3, aws.dynamodb)
    second_files = {**_partitions(aws, "games/run-1/"), **_partitions(aws, "reviews/run-1/")}
    assert second_files == first_files
    assert second.games_partitions == first.games_partitions
    assert second.new_game_ids == 0


@responses.activate
def test_review_partitions_are_weighted_by_known_totals(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    for appid in range(1, 7):
        aws.games.put_item(Item={"appid": appid, "status": "scraped", "attempts": 0})
    aws.cursors.put_item(Item={"appid": 1, "last_review_ts": 0, "total_reviews": 100_000})
    _mock_app_list([])
    run("run-4", settings, client, aws.s3, aws.dynamodb)
    parts = _partitions(aws, "reviews/run-4/")
    assert [1] in parts.values()  # the heavy game gets a worker to itself


@responses.activate
def test_uncapped_scraped_games_weigh_their_new_reviews_only(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    for appid in range(1, 7):
        aws.games.put_item(Item={"appid": appid, "status": "scraped", "attempts": 0})
    for appid in (1, 2):  # huge lifetime totals, but only a week of new reviews to fetch
        aws.cursors.put_item(Item={"appid": appid, "last_review_ts": 0, "total_reviews": 1_000_000})
    aws.games.put_item(Item={"appid": 9, "status": "pending", "attempts": 0})
    aws.games.update_item(
        Key={"appid": 9},
        UpdateExpression="SET recommendations = :r",
        ExpressionAttributeValues={":r": 100_000},
    )
    _mock_app_list([])
    run("run-7", settings, client, aws.s3, aws.dynamodb)
    parts = _partitions(aws, "reviews/run-7/")
    assert [9] in parts.values()  # the new game's full history outweighs both scraped games
    # weighted by their lifetime totals, 1 and 2 would each get a worker to themselves
    assert [1] not in parts.values() and [2] not in parts.values()


@responses.activate
def test_pending_backfills_weigh_more_and_request_an_etl_full_refresh(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    for appid in range(1, 8):
        aws.games.put_item(Item={"appid": appid, "status": "scraped", "attempts": 0})
    cursors = {
        1: {"total_reviews": 50, "oldest_review_ts": 10},  # pending
        2: {"total_reviews": 50, "oldest_review_ts": 10, "backfill_complete": True},
        3: {"total_reviews": 50},  # legacy, not seeded
    }
    for appid, fields in cursors.items():
        aws.cursors.put_item(Item={"appid": appid, "last_review_ts": 0, **fields})
    # a new game bigger than the cap starts its backfill on its first scrape
    aws.games.put_item(Item={"appid": 9, "status": "pending", "attempts": 0})
    aws.games.update_item(
        Key={"appid": 9},
        UpdateExpression="SET recommendations = :r",
        ExpressionAttributeValues={":r": 5000},
    )
    _mock_app_list([])
    capped = settings.model_copy(update={"max_reviews_per_game": 2000})

    result = run("run-5", capped, client, aws.s3, aws.dynamodb)

    assert result.backfill_game_ids == 2 and result.etl_full_refresh
    parts = _partitions(aws, "reviews/run-5/")
    assert [1] in parts.values() and [9] in parts.values()  # ~200 requests each

    off = capped.model_copy(update={"backfill_reviews_per_run": 0})
    result = run("run-6", off, client, aws.s3, aws.dynamodb)
    assert result.backfill_game_ids == 0 and not result.etl_full_refresh


@responses.activate
def test_handler_uses_env_api_key_and_validates_event(
    aws: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAW_BUCKET", "raw-steam-data-test")
    monkeypatch.setenv("PARTITIONS_BUCKET", PARTITIONS_BUCKET)
    monkeypatch.setenv("STEAM_API_KEY", "env-key")
    _mock_app_list([(1, 100)])
    out = lambda_handler.handler({"run_id": "exec-1", "ignored": True}, None)
    assert out["games_partitions"] == ["games/exec-1/appids-000.json"]
    assert "key=env-key" in responses.calls[0].request.url
    json.dumps(out)  # Step Functions needs a JSON-serialisable payload

    with pytest.raises(ValueError):
        lambda_handler.handler({"run_id": "bad/id"}, None)
