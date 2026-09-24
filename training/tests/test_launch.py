from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from steam_training import launch
from steam_training.config import LaunchSettings

SETTINGS = LaunchSettings(
    model_artifacts_bucket="model-artifacts-1",
    sagemaker_role_arn="arn:aws:iam::123456789012:role/steam-recsys-training",
    training_image_repository="123456789012.dkr.ecr.us-east-1.amazonaws.com/training",
)
SHA = "0123456789abcdef0123456789abcdef01234567"


def test_parse_env():
    assert launch.parse_env(["EPOCHS=10", "RECALL_KS=[30,50]"]) == {
        "EPOCHS": "10",
        "RECALL_KS": "[30,50]",
    }
    for bad in ["epochs=1", "EPOCHS", "MODEL_ID=x", "USE_SSM=false"]:
        with pytest.raises(ValueError):
            launch.parse_env([bad])


def test_job_request_uses_the_models_own_image():
    now = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)
    request = launch.training_job_request(SETTINGS, "promote", SHA, {"EPOCHS": "2"}, now)
    name = request["TrainingJobName"]
    assert name == "steam-recsys-promote-0123456789ab-20260924120000" and len(name) <= 63
    spec = request["AlgorithmSpecification"]
    assert spec["TrainingImage"].endswith(f"/training:{SHA}")
    assert spec["ContainerEntrypoint"] == ["python", "-m", "steam_training", "promote"]
    assert request["Environment"] == {
        "EPOCHS": "2",
        "USE_SSM": "true",
        "MODEL_ID": SHA,
        "MODEL_ARTIFACTS_BUCKET": "model-artifacts-1",
    }
    assert request["ResourceConfig"]["InstanceType"] == "ml.m5.2xlarge"


def test_image_tag_and_instance_type_overrides(monkeypatch):
    sagemaker = MagicMock()
    sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "Failed"}
    monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
    monkeypatch.setattr(launch.boto3, "client", lambda *a, **k: sagemaker)
    args = ["train", "--model-id", SHA, "--image-tag", f"{SHA}-cu128"]
    launch.main([*args, "--instance-type", "ml.g4dn.xlarge"])
    request = sagemaker.create_training_job.call_args.kwargs
    assert request["AlgorithmSpecification"]["TrainingImage"].endswith(f":{SHA}-cu128")
    assert request["ResourceConfig"]["InstanceType"] == "ml.g4dn.xlarge"
    assert request["Environment"]["MODEL_ID"] == SHA


def test_main_fails_when_the_job_fails(monkeypatch):
    sagemaker = MagicMock()
    sagemaker.describe_training_job.return_value = {
        "TrainingJobStatus": "Failed",
        "FailureReason": "boom",
    }
    monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
    monkeypatch.setattr(launch.boto3, "client", lambda *a, **k: sagemaker)
    assert launch.main(["train", "--model-id", SHA, "--env", "EPOCHS=1 BATCH_SIZE=8"]) == 1
    request = sagemaker.create_training_job.call_args.kwargs
    assert request["Environment"]["BATCH_SIZE"] == "8"


def test_main_writes_the_training_summary(monkeypatch, tmp_path, settings, source, aws_env):
    import boto3
    from moto import mock_aws

    from steam_training.artifacts import ArtifactStore
    from steam_training.pipeline import run_training

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="model-artifacts-1")
        trained = settings.model_copy(update={"model_id": SHA, "epochs": 1})
        run_training(trained, source, ArtifactStore("model-artifacts-1", s3))

        sagemaker = MagicMock()
        sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "Completed"}
        clients = {"sagemaker": sagemaker, "s3": s3}
        monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
        monkeypatch.setattr(launch.boto3, "client", lambda name, **k: clients[name])
        summary = tmp_path / "summary.md"
        assert launch.main(["train", "--model-id", SHA, "--summary", str(summary)]) == 0
        text = summary.read_text()
        assert f"### Model `{SHA}`" in text and "| model | warm |" in text


def test_invalid_model_id_is_rejected():
    with pytest.raises(SystemExit):
        launch.main(["train", "--model-id", "champion"])


def test_wait_for_job_stops_watching_at_the_deadline(monkeypatch):
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    ticks = iter(range(0, 10_000, 60))
    sagemaker = MagicMock()
    sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "InProgress"}
    job = launch.wait_for_job(sagemaker, "job", 300, clock=lambda: next(ticks))
    assert job["TrainingJobStatus"] == "InProgress"
    assert sagemaker.describe_training_job.call_count == 5

    sagemaker.describe_training_job.side_effect = [
        {"TrainingJobStatus": "InProgress"},
        {"TrainingJobStatus": "Completed"},
    ]
    assert launch.wait_for_job(sagemaker, "job", 300)["TrainingJobStatus"] == "Completed"


def test_still_running_job_is_reported_not_failed(monkeypatch, tmp_path):
    sagemaker = MagicMock()
    sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "InProgress"}
    monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
    monkeypatch.setattr(launch.boto3, "client", lambda *a, **k: sagemaker)
    summary = tmp_path / "summary.md"
    args = ["train", "--model-id", SHA, "--wait-minutes", "0", "--summary", str(summary)]
    assert launch.main(args) == 0
    assert "is still running" in summary.read_text()


def test_promote_refuses_an_untrained_model(monkeypatch, aws_env):
    import boto3
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="model-artifacts-1")
        sagemaker = MagicMock()
        clients = {"sagemaker": sagemaker, "s3": s3}
        monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
        monkeypatch.setattr(launch.boto3, "client", lambda name, **k: clients[name])
        assert launch.main(["promote", "--model-id", SHA]) == 1
        sagemaker.create_training_job.assert_not_called()
