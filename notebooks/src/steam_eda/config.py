"""Settings of the notebooks (env vars `EDA_*`, no SSM: this runs on a workstation only).

AWS credentials, when `EDA_DATA_DIR` is an S3 prefix, come from the usual boto3 chain
(`AWS_PROFILE=admin`, …); nothing here reads or stores keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODULE_DIR = Path(__file__).resolve().parents[2]


class EdaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDA_", extra="ignore", frozen=True)

    # Folder (or s3://bucket/prefix) holding the public dataset files (spec 16 layout):
    # `reviews*.parquet` (or `reviews/*.parquet`), `games.parquet`, `game_tags.parquet`.
    data_dir: str = str(MODULE_DIR / "data")
    aws_region: str = "us-east-1"
    # Share of users kept by the hash sample (whole histories): 5% ≈ 6M reviews, 2M users.
    sample_users: float = Field(0.05, gt=0, le=1)
    seed: int = 0
    # Text models of the embedding / sentiment sections: "auto" = MiniLM + SST-2 when the nlp
    # group is installed (first run downloads them from the Hugging Face hub), "lexical" =
    # TF-IDF only and no sentiment (CI, offline).
    text_models: Literal["auto", "lexical"] = "auto"
    # Parquet cache of slow results (embeddings, sentiment scores), gitignored.
    cache_dir: Path = MODULE_DIR / ".cache"
