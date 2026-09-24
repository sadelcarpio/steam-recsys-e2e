"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/etl/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the deployed ECS task), so local runs and
tests work from plain env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import boto3
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SSM_PREFIX = "/etl/"
S3_URI_PATTERN = r"^s3://[a-z0-9.\-]+(/.*)?$"
# Glue database names: <schema>_raw / _staging / _intermediate / _marts.
SCHEMA_PATTERN = r"^[a-z][a-z0-9_]{0,40}$"

DEFAULT_DBT_PROJECT_DIR = Path(__file__).resolve().parents[2] / "dbt"


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/etl/`, keyed by the env var name."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._values: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._values is not None:
            return self._values
        self._values = {}
        if os.environ.get("USE_SSM", "false").lower() != "true":
            return self._values
        ssm = boto3.client("ssm")
        for page in ssm.get_paginator("get_parameters_by_path").paginate(
            Path=SSM_PREFIX, WithDecryption=True
        ):
            for param in page["Parameters"]:
                self._values[param["Name"].removeprefix(SSM_PREFIX).lower()] = param["Value"]
        return self._values

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._load().get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        values = self._load()
        return {name: values[name] for name in self.settings_cls.model_fields if name in values}


class EtlSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    athena_work_group: str
    athena_s3_staging_dir: str = Field(pattern=S3_URI_PATTERN)
    iceberg_s3_data_dir: str = Field(pattern=S3_URI_PATTERN)
    # Raw sources are read from the Glue database <dbt_schema>_raw.
    dbt_schema: str = Field("steam", pattern=SCHEMA_PATTERN)
    aws_region: str = "us-east-1"

    dbt_project_dir: Path = DEFAULT_DBT_PROJECT_DIR
    dbt_threads: int = Field(4, ge=1, le=32)
    # Rebuild everything except the append-only lookups (ids stay stable).
    full_refresh: bool = False
    # OPTIMIZE + VACUUM each incremental Iceberg table after it is written.
    iceberg_maintenance: bool = True
    reviews_lookback_days: int = Field(3, ge=0)
    games_lookback_days: int = Field(3, ge=0)

    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, SsmSettingsSource(settings_cls)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for noisy in ("botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
