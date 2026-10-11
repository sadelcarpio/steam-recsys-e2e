from __future__ import annotations

import io
from types import SimpleNamespace

import boto3
import polars as pl
import pytest

from steam_ingestion.anonymize import hash_user_id, resolve_user_id_key
from steam_ingestion.anonymize_raw_reviews import anonymize_frame, migrate
from steam_ingestion.config import IngestionSettings

from .conftest import RAW_BUCKET, USER_KEY

STEAM_ID = 76561197960287930


def test_hash_is_a_stable_63_bit_keyed_id() -> None:
    # known vector: HMAC-SHA256(key, "76561197960287930")[:8] >> 1
    assert hash_user_id(STEAM_ID, USER_KEY) == 4559049361332948866
    ids = [hash_user_id(STEAM_ID + i, USER_KEY) for i in range(1000)]
    assert all(0 <= i < 2**63 for i in ids)  # a non-negative bigint everywhere
    assert len(set(ids)) == 1000
    assert hash_user_id(STEAM_ID, b"x" * 32) != hash_user_id(STEAM_ID, USER_KEY)


def test_key_from_the_secret(aws: SimpleNamespace, settings: IngestionSettings) -> None:
    boto3.client("secretsmanager").create_secret(
        Name="data-ingestion/user-id-hmac-key", SecretString="s" * 40
    )
    deployed = settings.model_copy(update={"user_id_hmac_key": None})
    assert resolve_user_id_key(deployed) == b"s" * 40
    assert resolve_user_id_key(settings) == USER_KEY  # local / test override


def test_short_key_is_rejected(settings: IngestionSettings) -> None:
    with pytest.raises(ValueError, match="32 characters"):
        resolve_user_id_key(IngestionSettings(**{**settings.model_dump(), "user_id_hmac_key": "x"}))


def _put(aws: SimpleNamespace, key: str, df: pl.DataFrame) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf)
    aws.s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=buf.getvalue())


def _get(aws: SimpleNamespace, key: str) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(aws.s3.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read()))


def _legacy(authors: list[int | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rec_id": list(range(len(authors))),
            "author_id": pl.Series(authors, dtype=pl.Int64),
            "appid": [10] * len(authors),
            "review": ["text"] * len(authors),
        }
    )


def test_frame_keeps_column_order_and_nulls() -> None:
    out = anonymize_frame(_legacy([STEAM_ID, None, STEAM_ID]), USER_KEY)
    assert out.columns == ["rec_id", "user_id", "appid", "review"]
    assert out["user_id"].to_list() == [hash_user_id(STEAM_ID, USER_KEY), None, out["user_id"][0]]


def test_migration_rewrites_legacy_files_once(aws: SimpleNamespace) -> None:
    _put(aws, "reviews/2026-09-01-000-0000.parquet", _legacy([STEAM_ID, STEAM_ID + 1]))
    hashed = anonymize_frame(_legacy([STEAM_ID + 2]), USER_KEY)
    _put(aws, "reviews/2026-10-10-000-0000.parquet", hashed)  # written by the new scraper
    _put(aws, "games/2026-09-01-000-0000.parquet", pl.DataFrame({"appid": [10]}))

    counts, left = migrate(aws.s3, RAW_BUCKET, USER_KEY, workers=2)

    assert (counts, left) == ({"anonymized": 1, "skipped": 1}, [])
    df = _get(aws, "reviews/2026-09-01-000-0000.parquet")
    assert "author_id" not in df.columns
    assert df["user_id"].to_list() == [hash_user_id(s, USER_KEY) for s in (STEAM_ID, STEAM_ID + 1)]
    assert df["rec_id"].to_list() == [0, 1] and df["review"].to_list() == ["text", "text"]
    # resumable / idempotent: a second run rewrites nothing
    assert migrate(aws.s3, RAW_BUCKET, USER_KEY) == ({"skipped": 2}, [])


def test_dry_run_uploads_nothing(aws: SimpleNamespace) -> None:
    _put(aws, "reviews/a.parquet", _legacy([STEAM_ID]))
    assert migrate(aws.s3, RAW_BUCKET, USER_KEY, dry_run=True) == ({"dry-run": 1}, [])
    assert "author_id" in _get(aws, "reviews/a.parquet").columns


def test_inconsistent_file_fails_the_run(aws: SimpleNamespace) -> None:
    _put(aws, "reviews/a.parquet", _legacy([STEAM_ID]).with_columns(user_id=pl.lit(1)))
    with pytest.raises(ValueError, match="both"):
        migrate(aws.s3, RAW_BUCKET, USER_KEY)
