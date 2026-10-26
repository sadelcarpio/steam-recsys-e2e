"""Settings of the Kaggle export: env > SSM `/etl/*` (when `USE_SSM=true`), as `EtlSettings`."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from steam_etl.config import S3_URI_PATTERN, SCHEMA_PATTERN, SsmSettingsSource

DATASET_PATTERN = r"^[A-Za-z0-9_-]+/[a-z0-9-]{3,50}$"


def default_run_id() -> str:
    return datetime.now(UTC).strftime("manual-%Y%m%dT%H%M%SZ")


class KaggleExportSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    athena_work_group: str
    dbt_schema: str = Field("steam", pattern=SCHEMA_PATTERN)
    aws_region: str = "us-east-1"

    # UNLOAD destination; each run writes under <prefix><run_id>/<table>/ (expired after 7 days).
    export_s3_prefix: str = Field(pattern=S3_URI_PATTERN)
    # The last publish (`KaggleExportState`), outside the expiring prefix.
    export_state_s3_uri: str = Field(pattern=S3_URI_PATTERN)
    run_id: str = Field(default_factory=default_run_id, pattern=r"^[A-Za-z0-9_.:-]{1,80}$")

    kaggle_export_enabled: bool = False
    kaggle_dataset: str = Field(pattern=DATASET_PATTERN)
    kaggle_token_secret_id: str = "etl/kaggle-api-token"
    # "Data files (c) Original Authors": the reviews and game data aren't ours to relicense.
    kaggle_license: str = "copyright-authors"
    # A publish re-uploads every file (no incremental versions): ~17 GB of egress each time.
    kaggle_min_days_between: int = Field(28, ge=0)
    kaggle_status_timeout_minutes: int = Field(60, ge=1)
    kaggle_status_poll_seconds: float = Field(30.0, gt=0)

    work_dir: str = "/tmp/kaggle"
    # Rows per published reviews file (~1.5 GB of Parquet).
    rows_per_file: int = Field(10_000_000, ge=1)
    athena_poll_seconds: float = Field(2.0, gt=0)

    log_level: str = "INFO"

    @property
    def raw_database(self) -> str:
        return f"{self.dbt_schema}_raw"

    @property
    def intermediate_database(self) -> str:
        return f"{self.dbt_schema}_intermediate"

    @property
    def marts_database(self) -> str:
        return f"{self.dbt_schema}_marts"

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
