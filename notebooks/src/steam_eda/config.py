"""Settings of the notebooks (env vars, no SSM: this runs on a workstation only).

AWS credentials come from the usual boto3 chain (`AWS_PROFILE=admin`, …); nothing here reads
or stores keys.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import boto3
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdaSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    aws_region: str = "us-east-1"
    # Glue database of the ETL marts (Iceberg): interactions, game_features, game_tags, …
    marts_database: str = "steam_marts"
    # Root of the raw scraped parquet (`<root>/reviews/`, `<root>/games/`, `<root>/game_tags/`).
    # Default: s3://raw-steam-data-<account-id>. A local directory works too (tests, offline).
    raw_root: str | None = None
    # Rows per batch when streaming a mart: bounds the memory of a filtered / sampled load.
    batch_rows: int = Field(250_000, ge=1)
    # Local parquet cache of slow results (`steam_eda.cache.cached`).
    cache_dir: Path = Path(__file__).resolve().parents[2] / ".cache"

    @cached_property
    def raw_location(self) -> str:
        if self.raw_root:
            return self.raw_root.rstrip("/")
        account = boto3.client("sts", region_name=self.aws_region).get_caller_identity()["Account"]
        return f"s3://raw-steam-data-{account}"
