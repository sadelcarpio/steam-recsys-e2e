import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from steam_training.config import LaunchSettings, TrainingSettings


def test_env_overrides_and_json_lists(monkeypatch):
    monkeypatch.setenv("MODEL_ARTIFACTS_BUCKET", "model-artifacts-1")
    monkeypatch.setenv("MODEL_ID", "deadbeef")
    monkeypatch.setenv("EPOCHS", "7")
    monkeypatch.setenv("RECALL_KS", "[10, 20]")
    monkeypatch.setenv("PRIMARY_K", "20")
    settings = TrainingSettings()
    assert settings.epochs == 7 and settings.recall_ks == [10, 20]


@pytest.mark.parametrize(
    "overrides",
    [{"primary_k": 7}, {"model_id": "champion"}, {"model_id": "bad id"}, {"epochs": 0}],
)
def test_invalid_settings(overrides):
    base = {"model_artifacts_bucket": "model-artifacts-1", "model_id": "abc"}
    with pytest.raises(ValidationError):
        TrainingSettings(**{**base, **overrides})


def test_ssm_is_read_only_when_enabled(monkeypatch, aws_env):
    with mock_aws():
        ssm = boto3.client("ssm", region_name="us-east-1")
        for name, value in {
            "MODEL_ARTIFACTS_BUCKET": "model-artifacts-9",
            "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/steam-recsys-training",
            "TRAINING_IMAGE_REPOSITORY": "123456789012.dkr.ecr.us-east-1.amazonaws.com/training",
            "EPOCHS": "3",
        }.items():
            ssm.put_parameter(Name=f"/training/{name}", Value=value, Type="String")

        monkeypatch.setenv("MODEL_ID", "abc")
        monkeypatch.setenv("USE_SSM", "false")
        with pytest.raises(ValidationError):
            TrainingSettings()

        monkeypatch.setenv("USE_SSM", "true")
        monkeypatch.setenv("EPOCHS", "9")  # env wins over SSM
        settings = TrainingSettings()
        assert settings.model_artifacts_bucket == "model-artifacts-9"
        assert settings.epochs == 9
        launch = LaunchSettings()
        assert launch.training_image_repository.endswith("/training")
        assert launch.instance_type == "ml.m5.2xlarge"
