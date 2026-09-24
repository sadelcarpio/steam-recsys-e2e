import boto3
import pytest
from pydantic import ValidationError

from steam_serving.config import ServingSettings


def test_defaults(monkeypatch):
    monkeypatch.delenv("USE_SSM", raising=False)
    settings = ServingSettings()
    assert settings.recommendations_table == "game-explainable-recommendations"
    assert (settings.default_limit, settings.max_limit) == (10, 30)


def test_ssm_then_env(aws, monkeypatch):
    ssm = boto3.client("ssm")
    ssm.put_parameter(Name="/serving/GAME_DETAILS_TABLE", Value="details-x", Type="String")
    ssm.put_parameter(Name="/serving/MAX_LIMIT", Value="20", Type="String")
    monkeypatch.setenv("USE_SSM", "true")
    monkeypatch.setenv("MAX_LIMIT", "25")
    settings = ServingSettings()
    assert settings.game_details_table == "details-x"
    assert settings.max_limit == 25  # env wins


def test_default_limit_within_max(monkeypatch):
    monkeypatch.delenv("USE_SSM", raising=False)
    with pytest.raises(ValidationError):
        ServingSettings(default_limit=20, max_limit=10)
    with pytest.raises(ValidationError):
        ServingSettings(max_limit=101)
