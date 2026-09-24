"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/serving/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the deployed Lambda), so local runs and tests
work from plain env vars. Settings are read once per Lambda container (cold start).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

SSM_PREFIX = "/serving/"
# BatchGetItem reads at most 100 keys per call: one call enriches a whole list.
MAX_LIMIT = 100


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/serving/`, keyed by the env var name."""

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


class ServingSettings(BaseSettings):
    """Settings of the `recsys-serving` Lambda."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    # Written by the inference pipeline (inference/src/steam_inference/contracts.py).
    recommendations_table: str = "game-explainable-recommendations"
    game_details_table: str = "game-details"
    aws_region: str = "us-east-1"  # the Lambda runtime sets AWS_REGION

    # Recommendations returned when the request has no `limit` / at most (inference writes 30).
    default_limit: int = Field(10, ge=1, le=MAX_LIMIT)
    max_limit: int = Field(30, ge=1, le=MAX_LIMIT)
    # Attach game details (description, image, ...) unless the request says `details=false`.
    include_details: bool = True
    # Cache-Control max-age of successful responses (recommendations change weekly).
    cache_max_age_seconds: int = Field(300, ge=0)

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

    @model_validator(mode="after")
    def _check(self) -> ServingSettings:
        if self.default_limit > self.max_limit:
            raise ValueError(
                f"default_limit={self.default_limit} must be <= max_limit={self.max_limit}"
            )
        return self


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(), format="%(levelname)s %(name)s %(message)s", force=True
    )
    for noisy in ("botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
