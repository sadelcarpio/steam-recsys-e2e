from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws

from steam_ingestion.config import IngestionSettings
from steam_ingestion.steam_api import SteamClient

RAW_BUCKET = "raw-steam-data-test"
PARTITIONS_BUCKET = "game-partitions-test"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("USE_SSM", raising=False)


@pytest.fixture
def settings() -> IngestionSettings:
    return IngestionSettings(
        raw_bucket=RAW_BUCKET,
        partitions_bucket=PARTITIONS_BUCKET,
        num_review_workers=3,
        games_per_task=4,
        games_flush_every=2,
        reviews_flush_rows=3,
        max_reviews_per_game=0,
        steam_api_key="test-key",
    )


@pytest.fixture
def aws() -> Iterator[SimpleNamespace]:
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=RAW_BUCKET)
        s3.create_bucket(Bucket=PARTITIONS_BUCKET)
        dynamodb = boto3.resource("dynamodb")
        for name in ("game-ids-state", "reviews-state-cursor"):
            dynamodb.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": "appid", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "appid", "AttributeType": "N"}],
                BillingMode="PAY_PER_REQUEST",
            )
        yield SimpleNamespace(
            s3=s3,
            dynamodb=dynamodb,
            games=dynamodb.Table("game-ids-state"),
            cursors=dynamodb.Table("reviews-state-cursor"),
        )


@pytest.fixture
def client() -> SteamClient:
    return SteamClient("test-key", request_interval=0, max_retries=2, sleep=lambda _: None)
