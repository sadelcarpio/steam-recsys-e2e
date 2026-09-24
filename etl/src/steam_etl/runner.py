"""Runs the dbt project in-process with the settings mapped onto profiles.yml / vars."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from steam_etl.config import EtlSettings

logger = logging.getLogger(__name__)


class DbtRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    success: bool
    failures: list[str]


def dbt_env(settings: EtlSettings) -> dict[str, str]:
    """Env vars read by dbt/profiles.yml and dbt_project.yml."""
    return {
        "AWS_REGION": settings.aws_region,
        "DBT_SCHEMA": settings.dbt_schema,
        "ATHENA_WORK_GROUP": settings.athena_work_group,
        "ATHENA_S3_STAGING_DIR": settings.athena_s3_staging_dir,
        "ICEBERG_S3_DATA_DIR": settings.iceberg_s3_data_dir,
        "DBT_THREADS": str(settings.dbt_threads),
    }


def dbt_vars(settings: EtlSettings) -> dict[str, object]:
    return {
        "reviews_lookback_days": settings.reviews_lookback_days,
        "games_lookback_days": settings.games_lookback_days,
        "iceberg_maintenance": settings.iceberg_maintenance,
    }


def dbt_args(
    settings: EtlSettings, command: str = "build", extra_args: Sequence[str] = ()
) -> list[str]:
    project_dir = str(settings.dbt_project_dir)
    args = [
        command,
        "--project-dir",
        project_dir,
        "--profiles-dir",
        project_dir,
        "--vars",
        json.dumps(dbt_vars(settings)),
    ]
    if settings.full_refresh and command in {"build", "run"}:
        args.append("--full-refresh")
    return [*args, *extra_args]


def run_dbt(
    settings: EtlSettings, command: str = "build", extra_args: Sequence[str] = ()
) -> DbtRunResult:
    # Imported lazily: dbt is heavy and pulls in its own logging setup.
    from dbt.cli.main import dbtRunner

    os.environ.update(dbt_env(settings))
    args = dbt_args(settings, command, extra_args)
    logger.info("dbt %s", " ".join(args))
    res = dbtRunner().invoke(args)

    failures: list[str] = []
    if res.exception is not None:
        failures.append(f"{type(res.exception).__name__}: {res.exception}")
    for node_result in getattr(res.result, "results", None) or []:
        if str(node_result.status) in {"error", "fail", "runtime error"}:
            failures.append(f"{node_result.node.unique_id}: {node_result.message}")
    return DbtRunResult(command=command, success=res.success, failures=failures)
