"""ECS task `tags-scraping`: the Steam user tags of every known game (spec 8).

Games: every appid in `game-ids-state` that is `scraped` or `pending` (pending games are real
apps from GetAppList, so the games scraped in the same run get their tags too). Tags change as
players vote, so the whole catalog is refreshed every run: GetItems in batches of
`tags_batch_size` ids.

Output: `s3://raw-steam-data-*/game_tags/<scrape-date>-<part>.parquet`, one row per game with at
least one tag, flushed every `tags_flush_every` games. Re-runs only add rows (the ETL keeps the
latest scrape per game). On SIGTERM (Spot interruption) it flushes and exits non-zero.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import boto3
import polars as pl

from steam_ingestion.config import IngestionSettings, configure_logging
from steam_ingestion.models import GameStatus, GameTagsRecord
from steam_ingestion.schemas import GAME_TAGS_SCHEMA
from steam_ingestion.shutdown import EXIT_CODE, Shutdown, ShutdownRequested
from steam_ingestion.state import GameIdsState
from steam_ingestion.steam_api import GameTag, SteamApiError, SteamClient
from steam_ingestion.storage import next_part_number, put_parquet

logger = logging.getLogger(__name__)

TAGGED_STATUSES = frozenset({GameStatus.SCRAPED, GameStatus.PENDING})


def games_to_tag(state: GameIdsState) -> list[int]:
    return sorted(a for a, s in state.load_all().items() if s.status in TAGGED_STATUSES)


def build_tags_record(
    appid: int, tags: list[GameTag], names: dict[int, str], scraped_at: int, scrape_date: date
) -> GameTagsRecord | None:
    """None when none of the tags has a known name."""
    named = [t for t in tags if t.tag_id in names]
    if not named:
        return None
    return GameTagsRecord(
        appid=appid,
        tag_ids=[t.tag_id for t in named],
        tag_names=[names[t.tag_id] for t in named],
        tag_weights=[t.weight for t in named],
        scraped_at=scraped_at,
        scrape_date=scrape_date,
    )


def scrape_tags(
    settings: IngestionSettings,
    client: SteamClient,
    s3: Any,
    dynamodb: Any,
    today: date | None = None,
    shutdown: Shutdown | None = None,
    clock: Callable[[], float] = time.time,
) -> bool:
    """Returns False when more than `max_failure_ratio` of the batches failed. Raises
    `ShutdownRequested` (after flushing) when `shutdown` was requested."""
    shutdown = shutdown or Shutdown()
    today = today or datetime.now(UTC).date()
    prefix = f"game_tags/{today.isoformat()}-"
    part = next_part_number(s3, settings.raw_bucket, prefix)

    appids = games_to_tag(GameIdsState(dynamodb.Table(settings.game_ids_table)))
    names = client.get_tag_list()
    size = settings.tags_batch_size
    batches = [appids[i : i + size] for i in range(0, len(appids), size)]
    logger.info(
        "tagging %d games in %d batches (%d tag names)", len(appids), len(batches), len(names)
    )

    buffer: list[GameTagsRecord] = []
    failures = written = 0
    stopped = False

    def flush() -> None:
        nonlocal part, written
        if not buffer:
            return
        df = pl.DataFrame([r.model_dump() for r in buffer], schema=GAME_TAGS_SCHEMA)
        key = f"{prefix}{part:04d}.parquet"
        put_parquet(s3, settings.raw_bucket, key, df)
        written += len(buffer)
        logger.info("wrote %d games to s3://%s/%s", len(buffer), settings.raw_bucket, key)
        buffer.clear()
        part += 1

    for i, batch in enumerate(batches, 1):
        try:
            shutdown.check()
            tags = client.get_game_tags(batch, settings.tags_per_game)
        except ShutdownRequested:
            stopped = True
            break
        except SteamApiError as exc:
            failures += 1
            logger.warning("batch %d (appids %d..%d) failed: %s", i, batch[0], batch[-1], exc)
            continue
        scraped_at = int(clock())
        for appid in batch:
            record = build_tags_record(appid, tags.get(appid, []), names, scraped_at, today)
            if record is not None:
                buffer.append(record)
        if len(buffer) >= settings.tags_flush_every:
            flush()
        if i % 200 == 0:
            logger.info("progress %d/%d batches (failures=%d)", i, len(batches), failures)
    flush()

    logger.info(
        "%s: %d games with tags of %d, %d/%d batches failed",
        "stopped" if stopped else "done",
        written,
        len(appids),
        failures,
        len(batches),
    )
    if stopped:
        raise ShutdownRequested
    return not batches or failures / len(batches) <= settings.max_failure_ratio


def main() -> int:
    settings = IngestionSettings()
    configure_logging(settings.log_level)
    shutdown = Shutdown()
    shutdown.install()
    client = SteamClient(
        request_interval=settings.tags_request_interval_seconds,
        max_retries=settings.max_retries,
        max_backoff=settings.max_backoff_seconds,
        throttle_cooldown=settings.throttle_cooldown_seconds,
        sleep=shutdown.sleep,
    )
    try:
        ok = scrape_tags(
            settings, client, boto3.client("s3"), boto3.resource("dynamodb"), shutdown=shutdown
        )
    except ShutdownRequested:
        return EXIT_CODE
    return 0 if ok else 1
