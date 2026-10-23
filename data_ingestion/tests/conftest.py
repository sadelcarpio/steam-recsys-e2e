from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from moto import mock_aws

from steam_ingestion.config import IngestionSettings
from steam_ingestion.steam_api import APP_REVIEWS_URL, SteamClient

RAW_BUCKET = "raw-steam-data-test"
PARTITIONS_BUCKET = "game-partitions-test"
USER_KEY = b"k" * 32  # HMAC key of the pseudonymous user ids (spec 13)
REVIEWS_URL = APP_REVIEWS_URL  # one URL for every app: the appid is in `input_json`


def reviews_request(**fields: Any) -> Callable[[Any], tuple[bool, str]]:
    """`responses` matcher on the GetAppReviews `input_json` (subset match). A field set to
    `None` must be absent."""

    def match(request: Any) -> tuple[bool, str]:
        sent = json.loads(request.params.get("input_json", "{}"))
        for key, value in fields.items():
            if key in sent if value is None else sent.get(key) != value:
                return False, f"input_json {key}={sent.get(key)!r}, expected {value!r}"
        return True, ""

    return match


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
        num_games_workers=1,
        games_per_task=4,
        games_flush_every=2,
        reviews_flush_rows=3,
        max_reviews_per_game=0,
        steam_api_key="test-key",
        user_id_hmac_key=USER_KEY.decode(),
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
