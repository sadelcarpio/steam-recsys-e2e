"""One-off: seed `stored_reviews` of the review cursors still backfilling (backfill cap).

    AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_stored_reviews [--dry-run]

Reads, per appid, the number of reviews already scraped (Athena over the raw reviews) and sets
`stored_reviews` on every cursor with a pending backfill that has no count yet. Without it the
cap (`BACKFILL_MAX_REVIEWS_PER_GAME`) does not apply to that game; complete cursors never need
it. The next run marks the games at or over the cap `backfill_complete`. Conditional writes:
re-running is a no-op. Run it with the scraper image that writes `stored_reviews` deployed and
while no pipeline execution is running (an older scraper would drop the field).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Callable

import boto3

from steam_ingestion.config import IngestionSettings, configure_logging
from steam_ingestion.seed_backfill import Scraped, query_scraped
from steam_ingestion.state import ReviewCursor, ReviewsCursorState

logger = logging.getLogger(__name__)


def seed(
    cursors: dict[int, ReviewCursor],
    scraped: dict[int, Scraped],
    write: Callable[[int, int], bool],
    max_stored: int,
) -> Counter[str]:
    """Seed every pending cursor without a count; counts `capped` (the next run completes it) /
    `pending` / `skipped` (not pending, already seeded, or written concurrently)."""
    counts: Counter[str] = Counter()
    for appid, cursor in cursors.items():
        if not cursor.backfill_pending or cursor.stored_reviews is not None:
            counts["skipped"] += 1
            continue
        stored = scraped[appid].reviews if appid in scraped else 0
        if not write(appid, stored):
            counts["skipped"] += 1
        elif max_stored and stored >= max_stored:
            counts["capped"] += 1
        else:
            counts["pending"] += 1
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
        help="cap to report against (BACKFILL_MAX_REVIEWS_PER_GAME)",
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
        return args.dry_run or state.seed_stored_reviews(appid, stored)

    counts = seed(cursors, scraped, write, args.max_stored)
    logger.info("%s%s", "dry run: " if args.dry_run else "", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
