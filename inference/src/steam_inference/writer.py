"""Stage 4: write one item per user (`contracts.UserRecommendations`).

`DynamoWriter` overwrites the items with parallel batch writes (each worker thread owns its
boto3 resource; `batch_writer` resends unprocessed items and botocore retries throttling).
`JsonlWriter` writes the same items to a local file for dry runs.
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


class Writer(Protocol):
    def write(self, items: Iterable[UserRecommendations]) -> int:
        """Write every item, return how many were written."""
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
                "dynamodb",
                region_name=self.region,
                config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
            )
            self._local.table = resource.Table(self.table)
        return self._local.table

    def _write_chunk(self, chunk: list[dict]) -> int:
        with self._table().batch_writer(overwrite_by_pkeys=["user_id"]) as batch:
            for item in chunk:
                batch.put_item(Item=item)
        return len(chunk)

    def write(self, items: Iterable[UserRecommendations]) -> int:
        written = 0
        pending: set[Future] = set()
        iterator = (item.to_item() for item in items)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while chunk := list(islice(iterator, CHUNK_ITEMS)):
                if len(pending) >= 2 * self.concurrency:  # bound the items held in memory
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    written += sum(f.result() for f in done)
                    log.info("written %d items to %s", written, self.table)
                pending.add(pool.submit(self._write_chunk, chunk))
            written += sum(f.result() for f in pending)
        log.info("written %d items to %s", written, self.table)
        return written


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def write(self, items: Iterable[UserRecommendations]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.path.open("w") as out:
            for item in items:
                out.write(json.dumps(item.to_item(), default=_json_default) + "\n")
                written += 1
        log.info("written %d items to %s", written, self.path)
        return written


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")
