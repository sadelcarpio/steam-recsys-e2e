"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/data-ingestion/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the deployed Lambda / ECS tasks), so local
runs and tests work from plain env vars.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from pydantic import Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SSM_PREFIX = "/data-ingestion/"

logger = logging.getLogger(__name__)


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/data-ingestion/`, keyed by the env var name."""

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


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    raw_bucket: str
    partitions_bucket: str
    game_ids_table: str = "game-ids-state"
    reviews_cursor_table: str = "reviews-state-cursor"

    steam_api_key_secret_id: str = "data-ingestion/steam-api-key"
    # Local-dev / test fallback: when set, Secrets Manager is never called.
    steam_api_key: SecretStr | None = None

    num_review_workers: int = Field(10, ge=1)
    games_per_task: int = Field(8000, ge=1)

    # Steam store endpoints throttle at roughly 200 requests / 5 min per IP.
    request_interval_seconds: float = Field(1.5, ge=0)
    max_retries: int = Field(5, ge=0)
    max_backoff_seconds: float = Field(120.0, ge=0)

    games_flush_every: int = Field(500, ge=1)
    reviews_flush_rows: int = Field(50_000, ge=1)
    # 0 = unlimited. Caps newest-first reviews fetched per game per run (bounds the backfill).
    max_reviews_per_game: int = Field(2000, ge=0)
    max_game_attempts: int = Field(3, ge=1)
    # A task exits non-zero when more than this share of its games hit exhausted retries.
    max_failure_ratio: float = Field(0.2, ge=0, le=1)

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


class ScrapeTaskSettings(BaseSettings):
    """Per-task input injected by the Step Functions Distributed Map as container env."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    partition_key: str


def resolve_steam_api_key(settings: IngestionSettings) -> str:
    if settings.steam_api_key is not None:
        return settings.steam_api_key.get_secret_value()
    secret = boto3.client("secretsmanager").get_secret_value(
        SecretId=settings.steam_api_key_secret_id
    )["SecretString"]
    try:
        parsed = json.loads(secret)
    except json.JSONDecodeError:
        return secret.strip()
    return parsed["api_key"] if isinstance(parsed, dict) else str(parsed)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    # boto/urllib3 are chatty at INFO and would dominate the log bill.
    for noisy in ("botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
