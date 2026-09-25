"""ECS task `reviews-scraping`: incremental reviews for one `reviews/<run_id>/part-<n>.json`.

For each appid, pages newest-first (`filter=recent`) until reaching the `last_review_ts`
stored in `reviews-state-cursor` (forward pass, capped by `max_reviews_per_game`). Then, while
older reviews are pending, it pages the range `[1, oldest_review_ts]` newest-first for up to
`backfill_reviews_per_run` reviews and moves `oldest_review_ts` back (backward pass); an
exhausted range marks the game `backfill_complete`. Rows are buffered and flushed to
`s3://raw-steam-data-*/reviews/<scrape-date>-<worker-id>-<part>.parquet`; after each flush the
cursors of games whose reviews are *fully* written are committed. A game split across a flush
keeps its old cursor, so a crash can only re-emit rows (dedupe on `rec_id` downstream), never
skip them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import boto3
import polars as pl

from steam_ingestion.config import IngestionSettings, ScrapeTaskSettings, configure_logging
from steam_ingestion.models import REVIEWS_PARTITION_RE, ReviewRecord
from steam_ingestion.schemas import REVIEWS_SCHEMA
from steam_ingestion.state import ReviewCursor, ReviewsCursorState
from steam_ingestion.steam_api import ReviewPage, SteamApiError, SteamClient
from steam_ingestion.storage import next_part_number, put_parquet, read_partition

logger = logging.getLogger(__name__)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _Walk:
    """What one pass over a game's review pages buffered."""

    count: int = 0
    oldest: int | None = None
    newest: int | None = None
    total: int | None = None  # `total_reviews` of the first page

    def add(self, ts: int) -> None:
        self.count += 1
        self.oldest = ts if self.oldest is None else min(self.oldest, ts)
        self.newest = ts if self.newest is None else max(self.newest, ts)


def build_review_record(appid: int, raw: dict[str, Any], scrape_date: date) -> ReviewRecord:
    author = raw.get("author") or {}
    score = raw.get("weighted_vote_score")
    return ReviewRecord(
        rec_id=int(raw["recommendationid"]),
        author_id=int(author["steamid"]),
        appid=appid,
        playtime_forever=_int(author.get("playtime_forever")),
        playtime_last_two_weeks=_int(author.get("playtime_last_two_weeks")),
        playtime_at_review=_int(author.get("playtime_at_review")),
        num_games_owned=_int(author.get("num_games_owned")),
        num_reviews=_int(author.get("num_reviews")),
        last_played=_int(author.get("last_played")),
        language=raw.get("language"),
        review=raw.get("review"),
        timestamp_created=int(raw["timestamp_created"]),
        timestamp_updated=_int(raw.get("timestamp_updated")),
        voted_up=raw.get("voted_up"),
        votes_up=_int(raw.get("votes_up")),
        votes_funny=_int(raw.get("votes_funny")),
        weighted_vote_score=float(score) if score not in (None, "") else None,
        comment_count=_int(raw.get("comment_count")),
        steam_purchase=raw.get("steam_purchase"),
        received_for_free=raw.get("received_for_free"),
        written_during_early_access=raw.get("written_during_early_access"),
        primarily_steam_deck=raw.get("primarily_steam_deck"),
        scrape_date=scrape_date,
    )


def scrape_partition(
    partition_key: str,
    settings: IngestionSettings,
    client: SteamClient,
    s3: Any,
    dynamodb: Any,
    today: date | None = None,
) -> bool:
    """Returns False when the failure ratio exceeded `max_failure_ratio`."""
    match = REVIEWS_PARTITION_RE.match(partition_key)
    if not match:
        raise ValueError(f"not a reviews partition key: {partition_key}")
    today = today or datetime.now(UTC).date()
    prefix = f"reviews/{today.isoformat()}-{match['n']}-"
    part = next_part_number(s3, settings.raw_bucket, prefix)

    appids = read_partition(s3, settings.partitions_bucket, partition_key).appids
    cursor_state = ReviewsCursorState(dynamodb.Table(settings.reviews_cursor_table), dynamodb)
    cursors = cursor_state.load(appids)
    logger.info("worker %s: %d games, %d with cursors", match["n"], len(appids), len(cursors))

    buffer: list[ReviewRecord] = []
    completed: dict[int, ReviewCursor] = {}  # finished games whose rows are all in `buffer`/S3
    failures = backfill_failures = backfill_rows = total_rows = 0

    def flush() -> None:
        nonlocal part, total_rows
        if buffer:
            df = pl.DataFrame([r.model_dump() for r in buffer], schema=REVIEWS_SCHEMA)
            key = f"{prefix}{part:04d}.parquet"
            put_parquet(s3, settings.raw_bucket, key, df)
            logger.info("wrote %d reviews to s3://%s/%s", len(buffer), settings.raw_bucket, key)
            total_rows += len(buffer)
            buffer.clear()
            part += 1
        if completed:
            cursor_state.save(completed)
            completed.clear()

    def collect(appid: int, pages: Iterable[ReviewPage], walk: _Walk) -> None:
        """Buffer every review of `pages` into `walk`, flushing when the buffer is full. On an
        error `walk` still describes what was buffered (and will be written) so far."""
        for page in pages:
            if walk.total is None:
                walk.total = page.total_reviews
            for raw in page.reviews:
                record = build_review_record(appid, raw, today)
                buffer.append(record)
                walk.add(record.timestamp_created)
            if len(buffer) >= settings.reviews_flush_rows:
                flush()  # current game is not in `completed`: its cursor stays put

    cap, budget = settings.max_reviews_per_game, settings.backfill_reviews_per_run
    for i, appid in enumerate(appids, 1):
        previous = cursors.get(appid)
        since = previous.last_review_ts if previous else 0
        total = previous.total_reviews if previous else None
        oldest = previous.oldest_review_ts if previous else None
        complete = previous.backfill_complete if previous else False

        # forward pass: reviews newer than the cursor
        forward = _Walk()
        try:
            collect(appid, client.iter_review_pages(appid, since, cap), forward)
        except SteamApiError as exc:
            failures += 1
            logger.warning("appid %d failed: %s", appid, exc)
        else:
            if forward.total is not None:
                total = forward.total
            if previous is None:
                # first scrape: the whole history unless the cap truncated the walk
                oldest, complete = forward.oldest, not (cap and forward.count >= cap)

            # backward pass: reviews older than the oldest one written (end_date is inclusive, so
            # the boundary second can repeat: the ETL dedupes on review_id)
            if budget and not complete and oldest is not None:
                backward = _Walk()
                try:
                    collect(appid, client.iter_review_pages(appid, 0, budget, oldest), backward)
                    complete = backward.count < budget  # the range ran out before the budget
                except SteamApiError as exc:
                    # keep what was buffered (the cursor moves back to it); the rest is next run's
                    backfill_failures += 1
                    logger.warning("appid %d backfill failed: %s", appid, exc)
                backfill_rows += backward.count
                if backward.oldest is not None:
                    oldest = min(oldest, backward.oldest)

            cursor = ReviewCursor(
                last_review_ts=max(since, forward.newest or 0),
                total_reviews=total,
                oldest_review_ts=oldest,
                backfill_complete=complete,
            )
            if cursor != previous:
                completed[appid] = cursor
        if i % 500 == 0:
            logger.info(
                "progress %d/%d (rows=%d, failures=%d)",
                i,
                len(appids),
                total_rows + len(buffer),
                failures,
            )
    flush()

    logger.info(
        "done: %d reviews (%d backfilled), %d failed games, %d failed backfills of %d",
        total_rows,
        backfill_rows,
        failures,
        backfill_failures,
        len(appids),
    )
    return not appids or failures / len(appids) <= settings.max_failure_ratio


def main() -> int:
    settings = IngestionSettings()
    task = ScrapeTaskSettings()
    configure_logging(settings.log_level)
    client = SteamClient(
        request_interval=settings.request_interval_seconds,
        max_retries=settings.max_retries,
        max_backoff=settings.max_backoff_seconds,
        throttle_cooldown=settings.throttle_cooldown_seconds,
    )
    ok = scrape_partition(
        task.partition_key, settings, client, boto3.client("s3"), boto3.resource("dynamodb")
    )
    return 0 if ok else 1
