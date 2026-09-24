"""ECS task `reviews-scraping`: incremental reviews for one `reviews/<run_id>/part-<n>.json`.

For each appid, pages newest-first (`filter=recent`) until reaching the `last_review_ts`
stored in `reviews-state-cursor`. Rows are buffered and flushed to
`s3://raw-steam-data-*/reviews/<scrape-date>-<worker-id>-<part>.parquet`; after each flush the
cursors of games whose reviews are *fully* written are committed. A game split across a flush
keeps its old cursor, so a crash can only re-emit rows (dedupe on `rec_id` downstream), never
skip them.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

import boto3
import polars as pl

from steam_ingestion.config import IngestionSettings, ScrapeTaskSettings, configure_logging
from steam_ingestion.models import REVIEWS_PARTITION_RE, ReviewRecord
from steam_ingestion.schemas import REVIEWS_SCHEMA
from steam_ingestion.state import ReviewCursor, ReviewsCursorState
from steam_ingestion.steam_api import SteamApiError, SteamClient
from steam_ingestion.storage import next_part_number, put_parquet, read_partition

logger = logging.getLogger(__name__)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    failures = total_rows = 0

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

    for i, appid in enumerate(appids, 1):
        previous = cursors.get(appid)
        since = previous.last_review_ts if previous else 0
        newest = since
        total = previous.total_reviews if previous else None
        try:
            for page in client.iter_review_pages(appid, since, settings.max_reviews_per_game):
                if page.total_reviews is not None:
                    total = page.total_reviews
                for raw in page.reviews:
                    record = build_review_record(appid, raw, today)
                    buffer.append(record)
                    newest = max(newest, record.timestamp_created)
                if len(buffer) >= settings.reviews_flush_rows:
                    flush()  # current game is not in `completed`: its cursor stays put
        except SteamApiError as exc:
            failures += 1
            logger.warning("appid %d failed: %s", appid, exc)
        else:
            if previous is None or newest != since or total != previous.total_reviews:
                completed[appid] = ReviewCursor(last_review_ts=newest, total_reviews=total)
        if i % 500 == 0:
            logger.info(
                "progress %d/%d (rows=%d, failures=%d)",
                i,
                len(appids),
                total_rows + len(buffer),
                failures,
            )
    flush()

    logger.info("done: %d reviews, %d failed games of %d", total_rows, failures, len(appids))
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
