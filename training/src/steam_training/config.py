"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/training/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the SageMaker jobs and by the CD launcher), so
local runs and tests work from plain env vars. Hyperparameters are plain env vars too: the
launcher forwards overrides (e.g. `EPOCHS=10`) as the job's environment.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import boto3
from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SSM_PREFIX = "/training/"
BUCKET_PATTERN = r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$"
# Models are keyed by the git commit sha of the training code (or any short slug locally).
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$"
CHAMPION = "champion"


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/training/`, keyed by the env var name."""

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


class _SsmSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

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


class TrainingSettings(_SsmSettings):
    """Settings of the training / promotion jobs (`python -m steam_training <mode>`)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    model_id: str = Field(pattern=MODEL_ID_PATTERN)
    # Glue database of the ETL marts (interactions, game_features, lkp_*).
    glue_database: str = "steam_marts"
    aws_region: str = "us-east-1"

    # ---- data ----
    # Last ~10% of the interactions (by time) are the validation split.
    validation_fraction: float = Field(0.1, gt=0, lt=0.5)
    # Share of empty-history (first positive review) rows kept for training. Inference never
    # sees an empty history; a few of them teach the model a "no history" fallback query.
    cold_row_fraction: float = Field(0.1, ge=0, le=1)
    # Probability of truncating a warm history (>= 2 games) to a random shorter prefix.
    history_dropout: float = Field(0.3, ge=0, le=1)
    # Probability of replacing a game_idx with OOV so the OOV row learns a generic item and
    # games without training interactions fall back to their content features.
    item_id_dropout: float = Field(0.1, ge=0, le=1)

    # ---- model ----
    game_embedding_dim: int = Field(64, ge=4)
    attribute_embedding_dim: int = Field(16, ge=2)
    hidden_dim: int = Field(128, ge=4)
    output_dim: int = Field(64, ge=4)

    # ---- optimisation ----
    epochs: int = Field(5, ge=1)
    batch_size: int = Field(1024, ge=2)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-6, ge=0)
    temperature: float = Field(0.05, gt=0)
    # Subtract log(sampling probability) from in-batch logits (popularity correction).
    logq_correction: bool = True
    seed: int = 42
    num_threads: int = Field(0, ge=0)  # 0 = torch default
    # auto = cuda when available (GPU image, `cu128` extra), else cpu
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Resume from s3://<bucket>/checkpoints/<model_id>/ when its fingerprint matches.
    resume: bool = True

    # ---- hard negatives ----
    # One of the user's negatively reviewed games (is_positive=false, training split).
    explicit_negatives: bool = True
    # One game sampled from the ranks [mine_skip_top, mine_skip_top + mine_pool_size) of the
    # current model, re-mined at the start of each epoch >= mining_start_epoch. The top ranks
    # are skipped because they are the likeliest false negatives.
    mined_negatives: bool = True
    mining_start_epoch: int = Field(1, ge=0)
    mine_skip_top: int = Field(5, ge=0)
    mine_pool_size: int = Field(50, ge=1)
    # Examples re-mined per epoch (a random subset when there are more; the others keep their
    # previous mined negative). Bounds mining time on large datasets.
    mine_max_rows: int = Field(2_000_000, ge=1)

    # ---- evaluation ----
    recall_ks: list[int] = Field(default_factory=lambda: [30, 50, 100], min_length=1)
    primary_k: int = 50
    # Rows of the validation split scored after each epoch (monitoring only); 0 = skip.
    epoch_eval_rows: int = Field(50_000, ge=0)
    eval_batch_size: int = Field(4096, ge=1)
    # Positive validation rows kept (uniform sample): ~500k gives recall within ~0.002.
    eval_max_rows: int = Field(500_000, ge=1)
    # Promotion: candidate must beat the champion's warm recall@primary_k by more than this.
    min_improvement: float = Field(0.0, ge=0)
    # Promotion: promote even when the candidate loses (the report keeps the real metrics and
    # why it would have been rejected). For deliberate swaps, e.g. a first full-data model.
    force_promotion: bool = False

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> TrainingSettings:
        if self.primary_k not in self.recall_ks:
            raise ValueError(f"primary_k={self.primary_k} must be one of recall_ks")
        if self.model_id == CHAMPION:
            raise ValueError(f"model_id '{CHAMPION}' is reserved")
        return self


# Settings that do not change what is trained (a checkpoint stays valid when they change).
NOT_FINGERPRINTED = {
    "model_artifacts_bucket",
    "model_id",
    "glue_database",
    "aws_region",
    "epochs",
    "num_threads",
    "device",
    "resume",
    "recall_ks",
    "primary_k",
    "epoch_eval_rows",
    "eval_batch_size",
    "eval_max_rows",
    "min_improvement",
    "force_promotion",
    "log_level",
}


class LaunchSettings(_SsmSettings):
    """Settings of the CD launcher that starts SageMaker jobs from GitHub Actions."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    sagemaker_role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/.+$")
    # ECR repository URL without tag; the launcher appends `:<model_id>`.
    training_image_repository: str
    instance_type: str = "ml.m5.2xlarge"
    volume_size_gb: int = Field(30, ge=1)
    max_runtime_seconds: int = Field(4 * 3600, ge=60)
    aws_region: str = "us-east-1"


class ExportSettings(_SsmSettings):
    """Settings of `python -m steam_training export` (numpy user tower of a saved model)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    log_level: str = "INFO"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for noisy in ("botocore", "boto3", "urllib3", "pyiceberg"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
