"""Stage 4: write one item per user (`contracts.UserRecommendations`), only when it changed.

`DynamoWriter`:
  - `stored_hashes` scans the table in parallel segments, reading `user_id` + `content_hash`
    only (~0.5 read units per item, a few cents per full scan);
  - `write` puts the items with parallel batch writes (each worker thread owns its boto3
    resource; `batch_writer` resends unprocessed items and botocore retries throttling);
  - `delete` removes users that no longer get recommendations.
`JsonlWriter` writes the items to a local file for dry runs (no stored state: writes them all).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config

from steam_inference.contracts import UserRecommendations

log = logging.getLogger(__name__)
CHUNK_ITEMS = 1000
RETRIES = Config(retries={"max_attempts": 10, "mode": "adaptive"})


class Writer(Protocol):
    def stored_hashes(self) -> dict[str, str]:
        """user_id -> content_hash of every stored item ("" when an item has no hash)."""
        ...

    def write(self, items: Iterable[UserRecommendations]) -> int:
        """Write (overwrite) every item, return how many were written."""
        ...

    def delete(self, user_ids: Iterable[str]) -> int:
        """Delete the items of these users, return how many were deleted."""
        ...


class DynamoWriter:
    def __init__(self, table: str, *, region: str, concurrency: int, session=None) -> None:
        self.table = table
        self.region = region
        self.concurrency = concurrency
        self.session_factory = session or boto3.session.Session
        self._local = threading.local()

    def _table(self):
        if not hasattr(self._local, "table"):
            resource = self.session_factory().resource(
                "dynamodb", region_name=self.region, config=RETRIES
            )
            self._local.table = resource.Table(self.table)
        return self._local.table

    # ---- read ----

    def _scan_segment(self, segment: int) -> dict[str, str]:
        client = self.session_factory().client("dynamodb", region_name=self.region, config=RETRIES)
        hashes: dict[str, str] = {}
        pages = client.get_paginator("scan").paginate(
            TableName=self.table,
            Segment=segment,
            TotalSegments=self.concurrency,
            ProjectionExpression="user_id, content_hash",
        )
        for page in pages:
            for item in page["Items"]:
                hashes[item["user_id"]["S"]] = item.get("content_hash", {}).get("S", "")
        return hashes

    def stored_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for segment in pool.map(self._scan_segment, range(self.concurrency)):
                hashes.update(segment)
        log.info("read %d stored items from %s", len(hashes), self.table)
        return hashes

    # ---- write ----

    def _put_chunk(self, chunk: list[dict]) -> int:
        with self._table().batch_writer(overwrite_by_pkeys=["user_id"]) as batch:
            for item in chunk:
                batch.put_item(Item=item)
        return len(chunk)

    def _delete_chunk(self, chunk: list[str]) -> int:
        with self._table().batch_writer(overwrite_by_pkeys=["user_id"]) as batch:
            for user_id in chunk:
                batch.delete_item(Key={"user_id": user_id})
        return len(chunk)

    def _run(self, work, chunks: Iterable[list], verb: str) -> int:
        """Apply `work` to every chunk on the thread pool, holding a bounded number of chunks."""
        done_count = 0
        pending: set[Future] = set()
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for chunk in chunks:
                if len(pending) >= 2 * self.concurrency:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    done_count += sum(f.result() for f in done)
                    log.info("%s %d items in %s", verb, done_count, self.table)
                pending.add(pool.submit(work, chunk))
            done_count += sum(f.result() for f in pending)
        log.info("%s %d items in %s", verb, done_count, self.table)
        return done_count

    def write(self, items: Iterable[UserRecommendations]) -> int:
        return self._run(self._put_chunk, _chunks(item.to_item() for item in items), "wrote")

    def delete(self, user_ids: Iterable[str]) -> int:
        return self._run(self._delete_chunk, _chunks(user_ids), "deleted")


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def stored_hashes(self) -> dict[str, str]:
        return {}

    def write(self, items: Iterable[UserRecommendations]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.path.open("w") as out:
            for item in items:
                out.write(json.dumps(item.to_item(), default=_json_default) + "\n")
                written += 1
        log.info("wrote %d items to %s", written, self.path)
        return written

    def delete(self, user_ids: Iterable[str]) -> int:
        return 0


def _chunks(values: Iterable) -> Iterable[list]:
    iterator = iter(values)
    while chunk := list(islice(iterator, CHUNK_ITEMS)):
        yield chunk


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")
