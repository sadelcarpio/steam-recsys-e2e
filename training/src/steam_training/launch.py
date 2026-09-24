"""Starts a SageMaker training job for `train` or `promote` and watches it (CD workflows).

    python -m steam_training.launch train   --model-id <sha> [--env EPOCHS=10 ...]
    python -m steam_training.launch promote --model-id <sha>

Both run the image `<training_image_repository>:<model_id>` by default, so a model is trained
and evaluated by the code of its own commit. `--image-tag` picks another image: the GPU build
(`<sha>-cu128`), or a pushed commit to promote a model trained locally (`python -m
steam_training train` with MODEL_ID=local-...). Settings come from env > SSM `/training/*`
(`USE_SSM=true`). With `--summary <file>` a Markdown summary of the result is appended to the file
(the workflows pass `$GITHUB_STEP_SUMMARY`).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

from steam_training.artifacts import ArtifactStore
from steam_training.config import MODEL_ID_PATTERN, LaunchSettings, configure_logging
from steam_training.contracts import EvaluationReport, ModelMetadata, RecallMetrics

log = logging.getLogger("steam_training.launch")

ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Set by the launcher itself; an override would detach the job from its model.
RESERVED_ENV = {"MODEL_ID", "USE_SSM", "MODEL_ARTIFACTS_BUCKET"}
TERMINAL = {"Completed", "Failed", "Stopped"}


def parse_env(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not ENV_NAME.match(name):
            raise ValueError(f"expected NAME=value with an upper-case NAME, got {pair!r}")
        if name in RESERVED_ENV:
            raise ValueError(f"{name} is set by the launcher")
        env[name] = value
    return env


def job_name(mode: str, model_id: str, now: datetime) -> str:
    # SageMaker: <= 63 chars, [a-zA-Z0-9-]
    return f"steam-recsys-{mode}-{model_id[:12]}-{now:%Y%m%d%H%M%S}".replace("_", "-")


def training_job_request(
    settings: LaunchSettings,
    mode: str,
    model_id: str,
    env: dict[str, str],
    now: datetime,
    image_tag: str | None = None,
) -> dict:
    return {
        "TrainingJobName": job_name(mode, model_id, now),
        "AlgorithmSpecification": {
            "TrainingImage": f"{settings.training_image_repository}:{image_tag or model_id}",
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "steam_training", mode],
        },
        "RoleArn": settings.sagemaker_role_arn,
        # Artifacts are written to models/ by the job itself; SageMaker only needs an output
        # path for its (empty) model.tar.gz.
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{settings.model_artifacts_bucket}/sagemaker/{mode}/"
        },
        "ResourceConfig": {
            "InstanceType": settings.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": settings.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": settings.max_runtime_seconds},
        "Environment": {
            **env,
            "USE_SSM": "true",
            "MODEL_ID": model_id,
            "MODEL_ARTIFACTS_BUCKET": settings.model_artifacts_bucket,
        },
        "Tags": [
            {"Key": "Project", "Value": "steam-recsys"},
            {"Key": "ModelId", "Value": model_id},
        ],
    }


def wait_for_job(
    sagemaker, name: str, timeout_seconds: float, poll_seconds: float = 60, clock=time.monotonic
) -> dict:
    """Poll until the job ends or `timeout_seconds` pass (then it is returned still running:
    the job itself keeps going, only the workflow stops watching it)."""
    deadline = clock() + timeout_seconds
    while True:
        job = sagemaker.describe_training_job(TrainingJobName=name)
        if job["TrainingJobStatus"] in TERMINAL or clock() >= deadline:
            return job
        log.info("%s: %s / %s", name, job["TrainingJobStatus"], job.get("SecondaryStatus"))
        time.sleep(poll_seconds)


# ---- summaries -----------------------------------------------------------------------------


def _metrics_table(rows: dict[str, RecallMetrics]) -> list[str]:
    ks = sorted(next(iter(rows.values())).warm.recall)
    lines = [
        "| model | segment | rows | " + " | ".join(f"recall@{k}" for k in ks) + " |",
        "|---|---|---|" + "---|" * len(ks),
    ]
    for label, metrics in rows.items():
        for segment in ("warm", "cold", "all"):
            seg = getattr(metrics, segment)
            values = " | ".join(f"{seg.recall[k]:.4f}" for k in ks)
            lines.append(f"| {label} | {segment} | {seg.rows} | {values} |")
    return lines


def training_summary(metadata: ModelMetadata) -> str:
    lines = [
        f"### Model `{metadata.model_id}`",
        f"Split cutoff `{metadata.split.cutoff}`: {metadata.split.train_rows} train / "
        f"{metadata.split.validation_rows} validation rows. "
        f"Epoch losses: {', '.join(f'{x:.4f}' for x in metadata.epoch_losses)}",
        "",
        *_metrics_table({"model": metadata.validation, "popularity": metadata.popularity_baseline}),
    ]
    return "\n".join(lines) + "\n"


def promotion_summary(report: EvaluationReport) -> str:
    rows = {f"candidate `{report.model_id[:12]}`": report.candidate.metrics}
    if report.champion is not None:
        rows[f"champion `{report.champion.model_id[:12]}`"] = report.champion.metrics
    rows["popularity"] = report.popularity_baseline
    verdict = "PROMOTED" if report.promoted else "not promoted"
    lines = [
        f"### Promotion of `{report.model_id}`: {verdict}",
        report.reason,
        "",
        f"Validation rows after `{report.split.cutoff}`.",
        "",
        *_metrics_table(rows),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steam_training.launch")
    parser.add_argument("mode", choices=["train", "promote"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--env", nargs="*", default=[], help="NAME=value job env overrides")
    parser.add_argument("--summary", type=Path, help="append a Markdown summary to this file")
    parser.add_argument(
        "--image-tag",
        help="image tag to run (default: the model id). E.g. `<sha>-cu128` for the GPU image, or "
        "a pushed commit's sha to promote a model trained locally",
    )
    parser.add_argument("--instance-type", help="override INSTANCE_TYPE (e.g. ml.g4dn.xlarge)")
    # GitHub OIDC credentials of the deploy role last 1 h: stop watching before they expire.
    parser.add_argument("--wait-minutes", type=float, default=50)
    args = parser.parse_args(argv)
    configure_logging("INFO")

    if not re.match(MODEL_ID_PATTERN, args.model_id) or args.model_id == "champion":
        parser.error(f"invalid --model-id {args.model_id!r}")
    settings = LaunchSettings()
    if args.instance_type:
        settings = settings.model_copy(update={"instance_type": args.instance_type})
    env = parse_env([pair for chunk in args.env for pair in chunk.split()])
    sagemaker = boto3.client("sagemaker", region_name=settings.aws_region)
    store = ArtifactStore(
        settings.model_artifacts_bucket, boto3.client("s3", region_name=settings.aws_region)
    )
    if args.mode == "promote" and store.load_metadata(args.model_id) is None:
        log.error("model %s is not trained: run the training CD first", args.model_id)
        return 1
    request = training_job_request(
        settings, args.mode, args.model_id, env, datetime.now(UTC), args.image_tag
    )
    name = request["TrainingJobName"]
    sagemaker.create_training_job(**request)
    log.info("started %s (%s)", name, settings.instance_type)

    job = wait_for_job(sagemaker, name, args.wait_minutes * 60)
    status = job["TrainingJobStatus"]
    log.info("%s: %s %s", name, status, job.get("FailureReason", ""))
    if status == "InProgress":
        summary = (
            f"### `{name}` is still running\n"
            f"Stopped watching after {args.wait_minutes:g} min; the job continues. Logs: "
            f"CloudWatch `/aws/sagemaker/TrainingJobs`, stream prefix `{name}`.\n"
        )
        _emit(summary, args.summary)
        return 0
    if status != "Completed":
        return 1

    if args.mode == "train":
        metadata = store.load_metadata(args.model_id)
        summary = training_summary(metadata) if metadata else "model metadata not found\n"
    else:
        report = store.read_report(args.model_id)
        summary = promotion_summary(report) if report else "evaluation report not found\n"
    _emit(summary, args.summary)
    return 0


def _emit(summary: str, path: Path | None) -> None:
    print(summary)
    if path:
        with path.open("a") as fh:
            fh.write(summary)


if __name__ == "__main__":
    sys.exit(main())
