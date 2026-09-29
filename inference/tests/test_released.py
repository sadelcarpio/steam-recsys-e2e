"""spec 10: unreleased (coming-soon) games are never published, like adult games."""

import gzip
from datetime import UTC, datetime

import boto3
import numpy as np
import pytest
from conftest import BUCKET, FIRST_GAME, N_GAMES, TABLE, FakeSource, details_table

from steam_inference.contracts import POPULAR_USER_ID, SearchIndex
from steam_inference.features import load_inference_data
from steam_inference.online import S3BundleStore
from steam_inference.pipeline import run_inference
from steam_inference.writer import DynamoWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)
# every third game is unreleased
UNRELEASED = [g for g in range(FIRST_GAME, FIRST_GAME + N_GAMES) if g % 3 == 1]


@pytest.fixture
def unreleased_marts(marts):
    games = range(FIRST_GAME, FIRST_GAME + N_GAMES)
    return {**marts, "game_details": details_table(games, coming_soon=UNRELEASED)}


def test_release_status_is_loaded_per_catalog_row(unreleased_marts):
    games = load_inference_data(FakeSource(unreleased_marts)).games
    expected = np.isin(games.catalog.items.game_idx, UNRELEASED)
    assert expected.any() and np.array_equal(games.coming_soon, expected)


def test_missing_game_details_counts_every_game_released(marts):
    without = {k: v for k, v in marts.items() if k != "game_details"}
    games = load_inference_data(FakeSource(without)).games
    assert not games.coming_soon.any()


def test_pipeline_publishes_nothing_unreleased(settings, unreleased_marts, store, champion):
    source = FakeSource(unreleased_marts)
    unreleased = {1000 + g for g in UNRELEASED}
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.unreleased_games == len(UNRELEASED)

    items = boto3.resource("dynamodb").Table(TABLE).scan()["Items"]
    recommended = {int(r["game_id"]) for i in items for r in i["recommendations"]}
    assert recommended and not recommended & unreleased
    popular = next(i for i in items if i["user_id"] == POPULAR_USER_ID)
    assert not {int(r["game_id"]) for r in popular["recommendations"]} & unreleased

    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=BUCKET, Key="serving/search/games.json")["Body"].read()
    searchable = {g for g, _, _ in SearchIndex.model_validate_json(gzip.decompress(body)).games}
    assert len(searchable) == summary.catalog_games - len(UNRELEASED)
    assert not searchable & unreleased
