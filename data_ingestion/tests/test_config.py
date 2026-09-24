from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from steam_ingestion.config import IngestionSettings, resolve_steam_api_key


def test_defaults_and_env_without_ssm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAW_BUCKET", "raw")
    monkeypatch.setenv("PARTITIONS_BUCKET", "parts")
    monkeypatch.setenv("NUM_REVIEW_WORKERS", "7")
    s = IngestionSettings()
    assert (s.raw_bucket, s.partitions_bucket, s.num_review_workers) == ("raw", "parts", 7)
    assert s.game_ids_table == "game-ids-state"


def test_ssm_parameters_are_loaded_and_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    with mock_aws():
        ssm = boto3.client("ssm")
        for name, value in {
            "RAW_BUCKET": "raw-from-ssm",
            "PARTITIONS_BUCKET": "parts-from-ssm",
            "NUM_REVIEW_WORKERS": "12",
        }.items():
            ssm.put_parameter(Name=f"/data-ingestion/{name}", Value=value, Type="String")
        monkeypatch.setenv("USE_SSM", "true")
        monkeypatch.setenv("NUM_REVIEW_WORKERS", "4")
        s = IngestionSettings()
    assert s.raw_bucket == "raw-from-ssm"
    assert s.partitions_bucket == "parts-from-ssm"
    assert s.num_review_workers == 4


def test_api_key_env_fallback_skips_secrets_manager() -> None:
    s = IngestionSettings(raw_bucket="r", partitions_bucket="p", steam_api_key="local")
    assert resolve_steam_api_key(s) == "local"


@pytest.mark.parametrize("secret", ["abc123", '{"api_key": "abc123"}'])
def test_api_key_from_secrets_manager(secret: str) -> None:
    with mock_aws():
        boto3.client("secretsmanager").create_secret(
            Name="data-ingestion/steam-api-key", SecretString=secret
        )
        s = IngestionSettings(raw_bucket="r", partitions_bucket="p")
        assert resolve_steam_api_key(s) == "abc123"
