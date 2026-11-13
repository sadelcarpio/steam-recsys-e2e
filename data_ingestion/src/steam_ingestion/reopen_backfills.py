"""One-off: reopen the review backfills that were marked complete by a walk that ended early.

    AWS_PROFILE=<admin> uv run python -m steam_ingestion.reopen_backfills [--dry-run]

Before the scraper checked Steam's count (`walk_exhausted`), a backward pass that came back short
of its budget marked the game `backfill_complete`, even when Steam's paging had just ended early
(on 2026-09-28/29 Rust stopped at 48k of 1.4M reviews). This reads, per appid, the reviews
already scraped (Athena over the raw reviews) and marks pending again every complete cursor that
has an `oldest_review_ts`, is under the cap (`BACKFILL_MAX_REVIEWS_PER_GAME`) and holds fewer
reviews than its Steam `total_reviews` by more than `unfetched_slack`. `stored_reviews` is seeded
from the same count when missing. Conditional writes: re-running is a no-op. Run it with the
scraper that checks Steam's count deployed (an older one would complete the games again) and
while no pipeline execution is running. The next runs then backfill these games (and rebuild the
marts with `etl_full_refresh`) until they reach the cap or their history ends.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Callable

import boto3

from steam_ingestion.config import IngestionSettings, configure_logging
from steam_ingestion.seed_backfill import Scraped, query_scraped
from steam_ingestion.state import ReviewCursor, ReviewsCursorState, walk_exhausted

logger = logging.getLogger(__name__)


def to_reopen(cursor: ReviewCursor, scraped: Scraped | None, max_stored: int) -> int | None:
    """The stored count to reopen `cursor` with, or None when it is really complete."""
    if not cursor.backfill_complete or cursor.oldest_review_ts is None:
        return None
    stored = scraped.reviews if scraped else 0
    counted = cursor.stored_reviews if cursor.stored_reviews is not None else stored
    if max_stored and counted >= max_stored:
        return None  # complete by the cap, not by the walk
    if walk_exhausted(stored, cursor.total_reviews):
        return None
    return stored


def reopen(
    cursors: dict[int, ReviewCursor],
    scraped: dict[int, Scraped],
    write: Callable[[int, int], bool],
    max_stored: int,
) -> Counter[str]:
    """Reopen every complete cursor that misses reviews; counts `reopened` (and the reviews
    `missing` from them, against Steam's total) / `skipped` (complete for good, not complete,
    or written concurrently)."""
    counts: Counter[str] = Counter()
    for appid, cursor in cursors.items():
        stored = to_reopen(cursor, scraped.get(appid), max_stored)
        if stored is None or not write(appid, stored):
            counts["skipped"] += 1
            continue
        counts["reopened"] += 1
        counts["missing"] += (cursor.total_reviews or 0) - stored
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", default="reviews-state-cursor")
    parser.add_argument("--work-group", default="steam-recsys-etl")
    parser.add_argument("--database", default="steam_raw", help="Glue database of the raw reviews")
    parser.add_argument(
        "--max-stored",
        type=int,
        default=IngestionSettings.model_fields["backfill_max_reviews_per_game"].default,
        help="cap (BACKFILL_MAX_REVIEWS_PER_GAME): games at it are complete",
    )
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args()
    configure_logging("INFO")

    dynamodb = boto3.resource("dynamodb")
    state = ReviewsCursorState(dynamodb.Table(args.table), dynamodb)
    cursors = state.load_all()
    scraped = query_scraped(boto3.client("athena"), args.work_group, args.database)
    logger.info("%d cursors, %d games with scraped reviews", len(cursors), len(scraped))

    def write(appid: int, stored: int) -> bool:
        if args.dry_run:
            logger.info("would reopen appid %d (%d stored)", appid, stored)
            return True
        return state.reopen_backfill(appid, stored)

    counts = reopen(cursors, scraped, write, args.max_stored)
    logger.info("%s%s", "dry run: " if args.dry_run else "", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
