"""S3 helpers shared by the Lambda and the scrapers (no polars import here: Lambda-safe)."""

from __future__ import annotations

import io
import re
from typing import Any

from steam_ingestion.models import PartitionFile


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def delete_prefix(s3: Any, bucket: str, prefix: str) -> None:
    keys = list_keys(s3, bucket, prefix)
    for i in range(0, len(keys), 1000):
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True},
        )


def write_partition(s3: Any, bucket: str, key: str, partition: PartitionFile) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=partition.model_dump_json().encode(),
        ContentType="application/json",
    )


def read_partition(s3: Any, bucket: str, key: str) -> PartitionFile:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return PartitionFile.model_validate_json(body)


def next_part_number(s3: Any, bucket: str, prefix: str) -> int:
    """First unused `<prefix><NNNN>.parquet` index, so task retries never overwrite output."""
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.parquet$")
    used = [int(m.group(1)) for k in list_keys(s3, bucket, prefix) if (m := pattern.match(k))]
    return max(used, default=-1) + 1


def put_parquet(s3: Any, bucket: str, key: str, df: Any) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
