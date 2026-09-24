import boto3
import pytest
from pydantic import ValidationError

from steam_inference.config import InferenceSettings


def test_reads_ssm_when_enabled(aws, monkeypatch):
    ssm = boto3.client("ssm")
    ssm.put_parameter(Name="/inference/MODEL_ARTIFACTS_BUCKET", Value="bucket-ssm", Type="String")
    ssm.put_parameter(Name="/inference/TOP_K", Value="25", Type="String")
    ssm.put_parameter(Name="/training/TOP_K", Value="99", Type="String")
    monkeypatch.setenv("USE_SSM", "true")
    monkeypatch.setenv("RERANK_MAX_USERS", "10")
    settings = InferenceSettings()
    assert settings.model_artifacts_bucket == "bucket-ssm"
    assert settings.top_k == 25
    assert settings.rerank_max_users == 10  # env
    monkeypatch.setenv("TOP_K", "40")
    assert InferenceSettings().top_k == 40  # env beats SSM


def test_ignores_ssm_by_default(monkeypatch):
    monkeypatch.delenv("USE_SSM", raising=False)
    monkeypatch.delenv("MODEL_ARTIFACTS_BUCKET", raising=False)
    with pytest.raises(ValidationError):
        InferenceSettings()


def test_explain_top_n_at_most_top_k(monkeypatch):
    monkeypatch.delenv("USE_SSM", raising=False)
    with pytest.raises(ValidationError, match="explain_top_n"):
        InferenceSettings(model_artifacts_bucket="bucket-x", top_k=3, explain_top_n=5)
