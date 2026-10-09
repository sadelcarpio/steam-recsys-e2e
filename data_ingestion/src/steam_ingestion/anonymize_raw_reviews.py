"""One-off (spec 13): replace the SteamID64 `author_id` of the raw review files written before
the scraper hashed it with the pseudonymous `user_id` (anonymize.py).

    ECS (in region, docs/deployment.md): aws ecs run-task --task-definition anonymize-raw-reviews
    local: AWS_PROFILE=<admin> USE_SSM=true uv run python -m steam_ingestion.anonymize_raw_reviews

Every `reviews/*.parquet` with an `author_id` column is rewritten in place: the column becomes
`user_id` (same position, hashed with the same key as the scraper) and the other columns are
untouched. The rewritten bytes are checked (same row count, no `author_id`) before the upload.
Files that already have `user_id` are skipped, so the task is idempotent and resumable. The raw
bucket is not versioned, so the overwrite removes the SteamIDs. It ends by checking every file
again and exits 1 when any still has `author_id`. Run it with the hashing scraper deployed and
no pipeline execution running (the schedule disabled).
"""

from __future__ import annotations

import argparse
import io
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
import polars as pl

from steam_ingestion.anonymize import hash_user_id, resolve_user_id_key
from steam_ingestion.config import IngestionSettings, configure_logging
from steam_ingestion.storage import list_keys

logger = logging.getLogger(__name__)

PREFIX = "reviews/"
RAW_ID, HASHED_ID = "author_id", "user_id"


def anonymize_frame(df: pl.DataFrame, key: bytes) -> pl.DataFrame:
    """`author_id` -> `user_id` in place (each distinct author hashed once; nulls stay null)."""
    authors = df[RAW_ID].drop_nulls().unique().to_list()
    mapping = {author: hash_user_id(author, key) for author in authors}
    return df.with_columns(
        pl.col(RAW_ID).replace_strict(mapping, default=None, return_dtype=pl.Int64)
    ).rename({RAW_ID: HASHED_ID})


def migrate_file(s3: Any, bucket: str, key: str, user_id_key: bytes, dry_run: bool) -> str:
    """`anonymized`, `skipped` (already hashed) or `dry-run`; raises on an inconsistent file."""
    df = pl.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))
    if RAW_ID not in df.columns:
        if HASHED_ID not in df.columns:
            raise ValueError(f"{key}: neither {RAW_ID} nor {HASHED_ID}")
        return "skipped"
    if HASHED_ID in df.columns:
        raise ValueError(f"{key}: both {RAW_ID} and {HASHED_ID}")
    out = anonymize_frame(df, user_id_key)
    buf = io.BytesIO()
    out.write_parquet(buf, compression="zstd")
    check = pl.read_parquet(io.BytesIO(buf.getvalue()))
    if RAW_ID in check.columns or check.height != df.height:
        raise RuntimeError(f"{key}: the rewritten file does not check out")
    if dry_run:
        return "dry-run"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return "anonymized"


def has_raw_ids(s3: Any, bucket: str, key: str) -> bool:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return RAW_ID in pl.read_parquet_schema(io.BytesIO(body))


def migrate(
    s3: Any, bucket: str, user_id_key: bytes, *, workers: int = 4, dry_run: bool = False
) -> tuple[Counter[str], list[str]]:
    """(counts per outcome, keys still holding raw ids after the run)."""
    keys = [k for k in list_keys(s3, bucket, PREFIX) if k.endswith(".parquet")]
    logger.info("%d review files under s3://%s/%s", len(keys), bucket, PREFIX)
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = pool.map(lambda k: migrate_file(s3, bucket, k, user_id_key, dry_run), keys)
        for i, outcome in enumerate(jobs, 1):
            counts[outcome] += 1
            if i % 100 == 0:
                logger.info("%d/%d files (%s)", i, len(keys), dict(counts))
        if dry_run:
            return counts, []
        flags = pool.map(lambda k: has_raw_ids(s3, bucket, k), keys)
        left = [k for k, raw in zip(keys, flags, strict=True) if raw]
    return counts, left


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=4, help="files processed in parallel")
    parser.add_argument("--dry-run", action="store_true", help="rewrite in memory, upload nothing")
    args = parser.parse_args()
    settings = IngestionSettings()
    configure_logging(settings.log_level)
    counts, left = migrate(
        boto3.client("s3"),
        settings.raw_bucket,
        resolve_user_id_key(settings),
        workers=args.workers,
        dry_run=args.dry_run,
    )
    logger.info("done: %s", dict(counts))
    if left:
        logger.error("%d files still hold %s, e.g. %s", len(left), RAW_ID, left[:5])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
