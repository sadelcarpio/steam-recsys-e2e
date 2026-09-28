from __future__ import annotations

import io
import json
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
import responses
from pydantic import ValidationError

from steam_ingestion.config import IngestionSettings
from steam_ingestion.models import GameTagsRecord
from steam_ingestion.schemas import GAME_TAGS_SCHEMA
from steam_ingestion.shutdown import Shutdown, ShutdownRequested
from steam_ingestion.state import GameIdsState
from steam_ingestion.steam_api import STORE_ITEMS_URL, TAG_LIST_URL, GameTag, SteamClient
from steam_ingestion.storage import list_keys
from steam_ingestion.tags_scraping.scraper import build_tags_record, games_to_tag, scrape_tags

from .conftest import RAW_BUCKET

TODAY = date(2026, 9, 28)
NAMES = {19: "Action", 1667: "Horror", 1742: "Story Rich"}


def _requested_appids(call: responses.Call) -> list[int]:
    query = json.loads(call.request.params["input_json"])
    return [i["appid"] for i in query["ids"]]


def _mock_steam(
    tags_by_appid: dict[int, list[tuple[int, int]]], failing: frozenset[int] = frozenset()
):
    responses.get(
        TAG_LIST_URL,
        json={"response": {"tags": [{"tagid": k, "name": v} for k, v in NAMES.items()]}},
    )

    def items(request):
        query = json.loads(request.params["input_json"])
        appids = [i["appid"] for i in query["ids"]]
        if failing & set(appids):
            return 503, {}, ""
        store_items = [
            {
                "appid": a,
                "id": a,
                "success": 1,
                "tags": [{"tagid": t, "weight": w} for t, w in tags],
            }
            if (tags := tags_by_appid.get(a)) is not None
            else {"id": a, "success": 15}
            for a in appids
        ]
        return 200, {}, json.dumps({"response": {"store_items": store_items}})

    responses.add_callback(responses.GET, STORE_ITEMS_URL, callback=items)


def _seed(aws: SimpleNamespace, statuses: dict[int, str]) -> None:
    for appid, status in statuses.items():
        aws.games.put_item(Item={"appid": appid, "status": status, "attempts": 0})
    aws.games.put_item(Item={"appid": 0, "last_modified": 1})  # catalog cursor


def _read(aws: SimpleNamespace) -> pl.DataFrame:
    frames = [
        pl.read_parquet(io.BytesIO(aws.s3.get_object(Bucket=RAW_BUCKET, Key=k)["Body"].read()))
        for k in sorted(list_keys(aws.s3, RAW_BUCKET, "game_tags/"))
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=GAME_TAGS_SCHEMA)


def test_games_to_tag_keeps_scraped_and_pending(aws: SimpleNamespace) -> None:
    _seed(aws, {1: "scraped", 2: "pending", 3: "unavailable", 4: "failed"})
    assert games_to_tag(GameIdsState(aws.games)) == [1, 2]


def test_build_tags_record_drops_unknown_tag_ids() -> None:
    tags = [GameTag(19, 900), GameTag(99999, 500), GameTag(1742, 100)]
    record = build_tags_record(7, tags, NAMES, 1_700_000_000, TODAY)
    assert record is not None
    assert record.tag_ids == [19, 1742]
    assert record.tag_names == ["Action", "Story Rich"]
    assert record.tag_weights == [900, 100]
    assert build_tags_record(7, [GameTag(99999, 1)], NAMES, 0, TODAY) is None
    assert build_tags_record(7, [], NAMES, 0, TODAY) is None


def test_tags_record_lists_must_align() -> None:
    with pytest.raises(ValidationError):
        GameTagsRecord(
            appid=1,
            tag_ids=[1, 2],
            tag_names=["a"],
            tag_weights=[1, 2],
            scraped_at=0,
            scrape_date=TODAY,
        )


@responses.activate
def test_scrape_tags_writes_one_row_per_tagged_game(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, {1: "scraped", 2: "pending", 3: "scraped", 4: "unavailable", 5: "scraped"})
    _mock_steam({1: [(1742, 100), (19, 900)], 2: [(1667, 50)], 3: []})
    settings = settings.model_copy(update={"tags_batch_size": 2, "tags_flush_every": 1})

    assert scrape_tags(settings, client, aws.s3, aws.dynamodb, today=TODAY, clock=lambda: 42.0)

    df = _read(aws).sort("appid")
    assert df.schema == GAME_TAGS_SCHEMA
    assert df["appid"].to_list() == [1, 2]  # 3 has no tags, 5 is unknown to Steam
    assert df["tag_names"].to_list() == [["Action", "Story Rich"], ["Horror"]]  # by weight
    assert df["tag_weights"].to_list() == [[900, 100], [50]]
    assert df["scraped_at"].to_list() == [42, 42]
    assert df["scrape_date"].to_list() == [TODAY, TODAY]
    # unavailable games are never requested; batches of 2 ids
    requested = [_requested_appids(c) for c in responses.calls if "GetItems" in c.request.url]
    assert requested == [[1, 2], [3, 5]]
    # flushed after the first batch; the second one had no tagged game
    assert list_keys(aws.s3, RAW_BUCKET, "game_tags/") == ["game_tags/2026-09-28-0000.parquet"]


@responses.activate
def test_retry_never_overwrites_previous_parts(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, {1: "scraped"})
    _mock_steam({1: [(19, 1)]})
    scrape_tags(settings, client, aws.s3, aws.dynamodb, today=TODAY)
    scrape_tags(settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert sorted(list_keys(aws.s3, RAW_BUCKET, "game_tags/")) == [
        "game_tags/2026-09-28-0000.parquet",
        "game_tags/2026-09-28-0001.parquet",
    ]


@responses.activate
@pytest.mark.parametrize(
    ("failing", "expected_ok"), [(frozenset({1}), True), (frozenset({1, 3}), False)]
)
def test_failed_batches_are_skipped_within_the_budget(
    aws: SimpleNamespace,
    settings: IngestionSettings,
    client: SteamClient,
    failing: frozenset[int],
    expected_ok: bool,
) -> None:
    # 5 batches of one game; max_failure_ratio 0.2 allows one failed batch
    _seed(aws, dict.fromkeys(range(1, 6), "scraped"))
    _mock_steam({a: [(19, 1)] for a in range(1, 6)}, failing=failing)
    settings = settings.model_copy(update={"tags_batch_size": 1})
    assert scrape_tags(settings, client, aws.s3, aws.dynamodb, today=TODAY) is expected_ok
    assert sorted(_read(aws)["appid"].to_list()) == sorted(set(range(1, 6)) - failing)


@responses.activate
def test_stop_flushes_and_raises(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, {1: "scraped", 2: "scraped"})
    _mock_steam({1: [(19, 1)], 2: [(19, 1)]})
    settings = settings.model_copy(update={"tags_batch_size": 1})
    shutdown = Shutdown()
    real = client.get_game_tags

    def tags_then_stop(appids: list[int], tag_count: int = 20):
        shutdown.request()  # SIGTERM during the first batch
        return real(appids, tag_count)

    client.get_game_tags = tags_then_stop  # type: ignore[method-assign]
    with pytest.raises(ShutdownRequested):
        scrape_tags(settings, client, aws.s3, aws.dynamodb, today=TODAY, shutdown=shutdown)
    assert _read(aws)["appid"].to_list() == [1]
