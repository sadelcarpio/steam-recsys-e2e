from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from steam_training import launch
from steam_training.config import LaunchSettings

# Built at import, before any fixture: the CD job sets USE_SSM=true for every step.
with pytest.MonkeyPatch.context() as mp:
    mp.delenv("USE_SSM", raising=False)
    SETTINGS = LaunchSettings(
        model_artifacts_bucket="model-artifacts-1",
        sagemaker_role_arn="arn:aws:iam::123456789012:role/steam-recsys-training",
        training_image_repository="123456789012.dkr.ecr.us-east-1.amazonaws.com/training",
    )
SHA = "0123456789abcdef0123456789abcdef01234567"
RealLogTail = launch.LogTail


@pytest.fixture(autouse=True)
def quiet_waits(monkeypatch):
    """No real sleeps, and main() gets a log tail that reads nothing (its own tests below use
    the real class)."""
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)

    class NoTail:
        def __init__(self, *a, **k):
            pass

        def poll(self):
            return 0

    monkeypatch.setattr(launch, "LogTail", NoTail)


def test_parse_env():
    assert launch.parse_env(["EPOCHS=10", "RECALL_KS=[30,50]"]) == {
        "EPOCHS": "10",
        "RECALL_KS": "[30,50]",
    }
    for bad in ["epochs=1", "EPOCHS", "MODEL_ID=x", "USE_SSM=false", "AWS_REGION=eu-west-1"]:
        with pytest.raises(ValueError):
            launch.parse_env([bad])
    with pytest.raises(ValueError):
        launch.parse_env(["GIT_COMMIT=x"])


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
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "MODEL_ID": SHA,
        "MODEL_ARTIFACTS_BUCKET": "model-artifacts-1",
        "GIT_COMMIT": SHA,
    }
    assert request["ResourceConfig"]["InstanceType"] == "ml.m5.2xlarge"
    names = {d["Name"] for d in spec["MetricDefinitions"]}
    assert names == set(launch.METRIC_DEFINITIONS["promote"])
    train = launch.training_job_request(SETTINGS, "train", SHA, {}, now)
    assert len(train["AlgorithmSpecification"]["MetricDefinitions"]) <= 40  # SageMaker limit


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
        clients = {"sagemaker": sagemaker, "s3": s3, "logs": MagicMock()}
        monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
        monkeypatch.setattr(launch.boto3, "client", lambda name, **k: clients[name])
        summary = tmp_path / "summary.md"
        assert launch.main(["train", "--model-id", SHA, "--summary", str(summary)]) == 0
        text = summary.read_text()
        assert f"### Model `{SHA}`" in text and "| model | warm |" in text


def test_invalid_model_id_is_rejected():
    with pytest.raises(SystemExit):
        launch.main(["train", "--model-id", "champion"])


def test_wait_for_job_stops_watching_at_the_deadline():
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
        clients = {"sagemaker": sagemaker, "s3": s3, "logs": MagicMock()}
        monkeypatch.setattr(launch, "LaunchSettings", lambda: SETTINGS)
        monkeypatch.setattr(launch.boto3, "client", lambda name, **k: clients[name])
        assert launch.main(["promote", "--model-id", SHA]) == 1
        sagemaker.create_training_job.assert_not_called()


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeLogs:
    """filter_log_events over a growing list of events, `page_size` per page, inclusive of
    `startTime` (as CloudWatch is)."""

    def __init__(self, page_size=2):
        self.events, self.page_size, self.calls = [], page_size, []
        self.error = None

    def add(self, timestamp, message):
        self.events.append(
            {"eventId": str(len(self.events)), "timestamp": timestamp, "message": message + "\n"}
        )

    def filter_log_events(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        matching = [e for e in self.events if e["timestamp"] >= kwargs["startTime"]]
        offset = int(kwargs.get("nextToken", 0))
        page = {"events": matching[offset : offset + self.page_size]}
        if offset + self.page_size < len(matching):
            page["nextToken"] = str(offset + self.page_size)
        return page


def test_log_tail_prints_each_line_once_across_pages_and_polls():
    logs, lines = FakeLogs(), []
    tail = RealLogTail(logs, "job", write=lines.append)
    for t, m in [(1, "loading"), (2, "a"), (2, "b")]:
        logs.add(t, m)
    assert tail.poll() == 3  # two pages
    logs.add(2, "c")  # same millisecond as the last line printed
    logs.add(5, "epoch 1/3")
    assert tail.poll() == 2 and tail.poll() == 0
    assert lines == ["loading", "a", "b", "c", "epoch 1/3"]
    assert logs.calls[0]["logStreamNamePrefix"] == "job/"
    assert logs.calls[-1]["startTime"] == 5


def test_log_tail_waits_for_the_stream_and_stops_on_other_errors(caplog):
    logs, lines = FakeLogs(), []
    tail = RealLogTail(logs, "job", write=lines.append)
    logs.error = ClientError("ResourceNotFoundException")  # the job hasn't logged yet
    assert tail.poll() == 0 and not tail.broken
    logs.error = ClientError("AccessDeniedException")
    assert tail.poll() == 0 and tail.broken
    logs.error = None
    logs.add(1, "never printed")
    assert tail.poll() == 0 and lines == []
    assert "stopped printing the job's logs" in caplog.text


def test_wait_for_job_prints_the_logs_until_the_end():
    logs, lines = FakeLogs(), []
    logs.add(1, "starting")
    sagemaker = MagicMock()
    statuses = iter(["InProgress", "InProgress", "Completed"])

    def describe(**kwargs):
        status = next(statuses)
        if status == "Completed":
            logs.add(9, "final_warm_recall=0.3")  # arrives with the last poll
        return {"TrainingJobStatus": status}

    sagemaker.describe_training_job.side_effect = describe
    tail = RealLogTail(logs, "job", write=lines.append)
    assert launch.wait_for_job(sagemaker, "job", 300, tail=tail)["TrainingJobStatus"] == "Completed"
    assert lines == ["starting", "final_warm_recall=0.3"]
