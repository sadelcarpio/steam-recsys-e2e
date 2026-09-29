"""One-off (spec 10): seed `coming_soon` of `game-ids-state` items scraped before spec 10.

    AWS_PROFILE=<admin> uv run python -m steam_ingestion.seed_coming_soon [--dry-run]

Reads the coming-soon games from the latest raw scrape of every appid (Athena), sets
`coming_soon = true` on their items (absent means false), and re-queues the ones that already
look released: a day-precise release date before today, or reviews already scraped. Their store
change happened before the Lambda watched for it, so GetAppList will not report it again.
Conditional writes: re-running is a no-op for seeded items, and an item already written by a
scrape is never touched. Run it while no pipeline execution is running.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import boto3

from steam_ingestion.config import configure_logging
from steam_ingestion.seed_backfill import run_query
from steam_ingestion.state import GameIdsState

logger = logging.getLogger(__name__)

QUERY = """
with latest as (
    select
        appid,
        coming_soon,
        release_date,
        row_number() over (partition by appid order by scrape_date desc) as scrape_rank
    from {database}.games
),
reviewed as (select distinct appid from {database}.reviews)
select l.appid, l.release_date, r.appid is not null as has_reviews
from latest as l
left join reviewed as r on r.appid = l.appid
where l.scrape_rank = 1 and l.coming_soon
"""

# Steam's day-precise formats (English store locale); anything else ("Q4 2026", "Coming soon",
# "2027", "October 2026") is not a release signal.
DATE_FORMATS = ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y")


@dataclass(frozen=True)
class ComingSoon:
    appid: int
    release_date: str | None
    has_reviews: bool


def parse_release_date(text: str | None) -> date | None:
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def looks_released(game: ComingSoon, today: date) -> bool:
    released = parse_release_date(game.release_date)
    return game.has_reviews or (released is not None and released < today)


def seed(
    games: list[ComingSoon],
    write: Callable[[int], bool],
    requeue: Callable[[int], bool],
    today: date,
    workers: int = 1,
) -> Counter[str]:
    """Seed every game; counts `seeded` / `skipped` (already set) and `requeued`."""

    def one(game: ComingSoon) -> list[str]:
        outcome = ["seeded" if write(game.appid) else "skipped"]
        if looks_released(game, today) and requeue(game.appid):
            outcome.append("requeued")
        return outcome

    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(one, games):
            counts.update(outcome)
    return counts


def query_coming_soon(athena: Any, work_group: str, database: str) -> list[ComingSoon]:
    rows = run_query(athena, QUERY.format(database=database), work_group)
    return [
        ComingSoon(appid=int(appid), release_date=release_date, has_reviews=has_reviews == "true")
        for appid, release_date, has_reviews in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", default="game-ids-state")
    parser.add_argument("--work-group", default="steam-recsys-etl")
    parser.add_argument("--database", default="steam_raw", help="Glue database of the raw data")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args()
    configure_logging("INFO")

    games = query_coming_soon(boto3.client("athena"), args.work_group, args.database)
    today = datetime.now(UTC).date()
    logger.info(
        "%d coming-soon games, %d look released",
        len(games),
        sum(looks_released(g, today) for g in games),
    )

    local = threading.local()

    def state() -> GameIdsState:
        if not hasattr(local, "state"):  # boto3 resources are not thread-safe: one per thread
            resource = boto3.session.Session().resource("dynamodb")
            local.state = GameIdsState(resource.Table(args.table))
        return local.state

    def write(appid: int) -> bool:
        return True if args.dry_run else state().seed_coming_soon(appid, True)

    def requeue(appid: int) -> bool:
        return True if args.dry_run else state().requeue([appid]) == 1

    counts = seed(games, write, requeue, today, args.workers)
    logger.info("%s%s", "dry run: " if args.dry_run else "", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
