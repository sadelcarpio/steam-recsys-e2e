"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/inference/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the ECS task), so local runs and tests work
from plain env vars.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

SSM_PREFIX = "/inference/"
BUCKET_PATTERN = r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$"


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/inference/`, keyed by the env var name."""

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


class InferenceSettings(BaseSettings):
    """Settings of the batch inference task (`python -m steam_inference`)."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    # Model served: the promoted champion, or any trained models/<model_id>/ (manual runs).
    model_id: str = "champion"
    # Glue database of the ETL marts (user_features, game_features, interactions, lkp_*).
    glue_database: str = "steam_marts"
    aws_region: str = "us-east-1"
    recommendations_table: str = "game-explainable-recommendations"
    # Details of every catalog game for serving (insert-only: only new games are written).
    game_details_table: str = "game-details"
    sync_game_details: bool = True
    # Online bundle for serving (s3://<model_artifacts_bucket>/<prefix>/bundle.npz + manifest).
    online_bundle_enabled: bool = True
    online_bundle_prefix: str = "serving/online"
    # Local file: write the items as JSON lines there instead of DynamoDB (dry runs).
    output_path: str | None = None

    # ---- retrieval ----
    # Candidates kept per user (all of them are written; the LLM reranks them).
    top_k: int = Field(30, ge=1, le=200)
    user_batch_size: int = Field(1024, ge=1)
    item_batch_size: int = Field(4096, ge=1)
    num_threads: int = Field(0, ge=0)  # 0 = torch default
    # Only the N most active users (0 = everyone with user_features); for local runs.
    max_users: int = Field(0, ge=0)
    # Popularity fallback (item "__popular__"): positive reviews in this many days before the
    # newest review.
    popular_window_days: int = Field(90, ge=1)

    # ---- LLM reranking (Bedrock Converse API) ----
    rerank_enabled: bool = True
    # Amazon Nova 2 Lite (US cross-region inference profile): ~1.8k input + ~0.4k output tokens
    # per reranked user. Any Converse model with tool use works (README: model comparison), e.g.
    # us.anthropic.claude-haiku-4-5-20251001-v1:0 (better explanations, ~2.5x the cost).
    bedrock_model_id: str = "us.amazon.nova-2-lite-v1:0"
    # Reranked users: at least this many reviews ("more than 5"), the most active first...
    rerank_min_reviews: int = Field(6, ge=1)
    # ... and at most this many per run (bounds the LLM cost).
    rerank_max_users: int = Field(1000, ge=0)
    # Recommendations that get an explanation (the top of the reranked list).
    explain_top_n: int = Field(5, ge=0)
    rerank_concurrency: int = Field(8, ge=1)
    rerank_max_tokens: int = Field(2000, ge=100)
    rerank_temperature: float = Field(0.2, ge=0, le=1)

    # ---- output ----
    # Parallel DynamoDB scan segments / batch writers. Only changed items are written; stored
    # users without recommendations are deleted (full runs only, i.e. MAX_USERS=0).
    write_concurrency: int = Field(8, ge=1)

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
    def _check(self) -> InferenceSettings:
        if self.explain_top_n > self.top_k:
            raise ValueError(f"explain_top_n={self.explain_top_n} must be <= top_k={self.top_k}")
        return self


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for noisy in ("botocore", "boto3", "urllib3", "pyiceberg"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
