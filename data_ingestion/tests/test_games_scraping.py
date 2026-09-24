from __future__ import annotations

import io
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
import responses

from steam_ingestion.config import IngestionSettings
from steam_ingestion.games_scraping.scraper import (
    build_game_record,
    parse_languages,
    scrape_partition,
)
from steam_ingestion.models import PartitionFile
from steam_ingestion.schemas import GAMES_SCHEMA
from steam_ingestion.steam_api import APP_DETAILS_URL, SteamClient
from steam_ingestion.storage import list_keys, write_partition

from .conftest import PARTITIONS_BUCKET, RAW_BUCKET

TODAY = date(2026, 9, 24)
KEY = "games/run-1/appids-000.json"

APP_DATA = {
    "type": "game",
    "name": "Portal",
    "required_age": "0",
    "is_free": False,
    "controller_support": "full",
    "detailed_description": "<p>detail</p>",
    "about_the_game": "about",
    "short_description": "short",
    "supported_languages": "English<strong>*</strong>, French, Simplified Chinese"
    "<br><strong>*</strong>languages with full audio support",
    "header_image": "https://img/header.jpg",
    "pc_requirements": {"minimum": "<ul>min</ul>", "recommended": "<ul>rec</ul>"},
    "developers": ["Valve"],
    "publishers": ["Valve"],
    "price_overview": {"currency": "USD", "initial": 999, "final": 199},
    "categories": [{"id": 2, "description": "Single-player"}],
    "genres": [{"id": "1", "description": "Action"}],
    "platforms": {"windows": True, "mac": True, "linux": False},
    "release_date": {"coming_soon": False, "date": "Oct 10, 2007"},
    "recommendations": {"total": 12345},
    "dlc": [323180],
}
SUMMARY = {"review_score": 9, "review_score_desc": "Overwhelmingly Positive"}


def _read_all(aws: SimpleNamespace, prefix: str) -> pl.DataFrame:
    frames = [
        pl.read_parquet(io.BytesIO(aws.s3.get_object(Bucket=RAW_BUCKET, Key=k)["Body"].read()))
        for k in sorted(list_keys(aws.s3, RAW_BUCKET, prefix))
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=GAMES_SCHEMA)


def _mock_details(appid: int, *, data: dict | None = None, success: bool = True) -> None:
    body = {str(appid): {"success": success, **({"data": data} if data else {})}}
    responses.get(
        APP_DETAILS_URL,
        match=[responses.matchers.query_param_matcher({"appids": str(appid)}, strict_match=False)],
        json=body,
    )
    responses.get(
        f"https://store.steampowered.com/appreviews/{appid}",
        json={"success": 1, "query_summary": SUMMARY},
    )


def _seed(aws: SimpleNamespace, appids: list[int]) -> None:
    for a in appids:
        aws.games.put_item(Item={"appid": a, "status": "pending", "attempts": 0})
    write_partition(aws.s3, PARTITIONS_BUCKET, KEY, PartitionFile(run_id="run-1", appids=appids))


def test_parse_languages() -> None:
    assert parse_languages(APP_DATA["supported_languages"]) == [
        "English",
        "French",
        "Simplified Chinese",
    ]
    assert parse_languages(None) == []


def test_build_game_record_maps_every_field() -> None:
    record = build_game_record(400, APP_DATA, SUMMARY, TODAY)
    assert record.price == 1.99
    assert record.required_age == 0
    assert record.minimum_pc_requirements == "<ul>min</ul>"
    assert record.categories == ["Single-player"] and record.genres == ["Action"]
    assert (record.windows_support, record.mac_support, record.linux_support) == (True, True, False)
    assert record.release_date == "Oct 10, 2007" and record.coming_soon is False
    assert record.recommendations == 12345 and record.dlc == [323180]
    assert record.review_score == 9
    df = pl.DataFrame([record.model_dump()], schema=GAMES_SCHEMA)
    assert df.schema == GAMES_SCHEMA


def test_build_game_record_tolerates_sparse_payload() -> None:
    record = build_game_record(1, {"name": "X", "is_free": True, "pc_requirements": []}, {}, TODAY)
    assert record.price == 0.0
    assert record.minimum_pc_requirements is None
    assert record.developers == [] and record.dlc == []


@responses.activate
def test_scrapes_partition_writes_parquet_and_marks_state(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, [1, 2, 3])
    _mock_details(1, data=APP_DATA)
    _mock_details(2, success=False)
    _mock_details(3, data={**APP_DATA, "name": "Portal 2"})

    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    df = _read_all(aws, "games/2026-09-24-000-")
    assert df.schema == GAMES_SCHEMA
    assert sorted(df["appid"].to_list()) == [1, 3]
    assert aws.games.get_item(Key={"appid": 1})["Item"]["status"] == "scraped"
    assert int(aws.games.get_item(Key={"appid": 1})["Item"]["recommendations"]) == 12345
    assert aws.games.get_item(Key={"appid": 2})["Item"]["status"] == "unavailable"


@responses.activate
def test_retry_never_overwrites_previous_parts(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, [1, 2, 3])
    for a in (1, 2, 3):
        _mock_details(a, data=APP_DATA)
    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    keys = sorted(list_keys(aws.s3, RAW_BUCKET, "games/"))
    # games_flush_every=2 -> 2 parts per run, numbering continues on the retry.
    assert keys == [f"games/2026-09-24-000-{n:04d}.parquet" for n in range(4)]


@responses.activate
@pytest.mark.parametrize(("n_failing", "expected_ok"), [(1, True), (3, False)])
def test_failure_budget(
    aws: SimpleNamespace,
    settings: IngestionSettings,
    client: SteamClient,
    n_failing: int,
    expected_ok: bool,
) -> None:
    appids = list(range(1, 6))
    _seed(aws, appids)
    for a in appids:
        if a <= n_failing:
            responses.get(
                APP_DETAILS_URL,
                match=[
                    responses.matchers.query_param_matcher({"appids": str(a)}, strict_match=False)
                ],
                status=503,
            )
        else:
            _mock_details(a, data=APP_DATA)
    ok = scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert ok is expected_ok
    item = aws.games.get_item(Key={"appid": 1})["Item"]
    assert item["status"] == "pending" and int(item["attempts"]) == 1


@responses.activate
def test_game_fails_permanently_after_max_attempts(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, [1])
    aws.games.update_item(
        Key={"appid": 1},
        UpdateExpression="SET attempts = :a",
        ExpressionAttributeValues={":a": settings.max_game_attempts - 1},
    )
    responses.get(APP_DETAILS_URL, status=503)
    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert aws.games.get_item(Key={"appid": 1})["Item"]["status"] == "failed"


def test_rejects_foreign_partition_key(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    with pytest.raises(ValueError):
        scrape_partition("reviews/run-1/part-000.json", settings, client, aws.s3, aws.dynamodb)
