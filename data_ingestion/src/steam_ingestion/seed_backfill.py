"""One-off: seed the backfill fields of `reviews-state-cursor` items written before the backfill.

    AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_backfill [--dry-run]

Reads, per appid, the oldest review and the number of reviews already scraped (Athena over the
raw reviews), then sets `oldest_review_ts` and `backfill_complete` (scraped count >= Steam's
`total_reviews`) on every cursor that has neither. Unseeded cursors are never backfilled.
Conditional writes: re-running is a no-op, and a cursor already written by the backfilling
scraper is never touched. Run it while no pipeline execution is running.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import boto3

from steam_ingestion.config import configure_logging
from steam_ingestion.state import ReviewCursor, ReviewsCursorState

logger = logging.getLogger(__name__)

QUERY = (
    "select appid, min(timestamp_created) as oldest, count(distinct rec_id) as reviews "
    "from {database}.reviews group by appid"
)


@dataclass(frozen=True)
class Scraped:
    oldest: int
    reviews: int


def backfill_fields(cursor: ReviewCursor, scraped: Scraped | None) -> tuple[int | None, bool]:
    """(oldest_review_ts, backfill_complete) of a cursor written before the backfill."""
    if scraped is None:
        # the game was scraped but Steam returned no review for it: nothing older to fetch
        return None, True
    return scraped.oldest, scraped.reviews >= (cursor.total_reviews or 0)


def seed(
    cursors: dict[int, ReviewCursor],
    scraped: dict[int, Scraped],
    write: Callable[[int, int | None, bool], bool],
    workers: int = 1,
) -> Counter[str]:
    """Seed every cursor without backfill fields; counts `pending` / `complete` / `skipped`
    (already seeded, or written concurrently)."""
    todo = {
        appid: backfill_fields(cursor, scraped.get(appid))
        for appid, cursor in cursors.items()
        if cursor.oldest_review_ts is None and not cursor.backfill_complete
    }
    counts: Counter[str] = Counter(skipped=len(cursors) - len(todo))

    def one(item: tuple[int, tuple[int | None, bool]]) -> str:
        appid, (oldest, complete) = item
        if not write(appid, oldest, complete):
            return "skipped"
        return "complete" if complete else "pending"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts.update(pool.map(one, todo.items()))
    return counts


def query_scraped(athena: Any, work_group: str, database: str) -> dict[int, Scraped]:
    execution = athena.start_query_execution(
        QueryString=QUERY.format(database=database), WorkGroup=work_group
    )["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution)["QueryExecution"]["Status"]
        if status["State"] == "SUCCEEDED":
            break
        if status["State"] in {"FAILED", "CANCELLED"}:
            raise RuntimeError(f"Athena query {status['State']}: {status.get('StateChangeReason')}")
        time.sleep(2)
    scraped: dict[int, Scraped] = {}
    pages = athena.get_paginator("get_query_results").paginate(QueryExecutionId=execution)
    for page_no, page in enumerate(pages):
        rows = page["ResultSet"]["Rows"]
        for row in rows[1:] if page_no == 0 else rows:  # the first row is the header
            appid, oldest, reviews = (c.get("VarCharValue") for c in row["Data"])
            scraped[int(appid)] = Scraped(oldest=int(oldest), reviews=int(reviews))
    return scraped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", default="reviews-state-cursor")
    parser.add_argument("--work-group", default="steam-recsys-etl")
    parser.add_argument("--database", default="steam_raw", help="Glue database of the raw reviews")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args()
    configure_logging("INFO")

    dynamodb = boto3.resource("dynamodb")
    cursors = ReviewsCursorState(dynamodb.Table(args.table), dynamodb).load_all()
    scraped = query_scraped(boto3.client("athena"), args.work_group, args.database)
    logger.info("%d cursors, %d games with scraped reviews", len(cursors), len(scraped))

    local = threading.local()

    def write(appid: int, oldest: int | None, complete: bool) -> bool:
        if args.dry_run:
            return True
        if not hasattr(local, "state"):  # boto3 resources are not thread-safe: one per thread
            resource = boto3.session.Session().resource("dynamodb")
            local.state = ReviewsCursorState(resource.Table(args.table), resource)
        return local.state.seed_backfill(appid, oldest, complete)

    counts = seed(cursors, scraped, write, args.workers)
    logger.info("%s%s", "dry run: " if args.dry_run else "", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
