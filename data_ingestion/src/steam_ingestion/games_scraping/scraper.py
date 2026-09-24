"""ECS task `games-scraping`: game details for one `games/<run_id>/appids-<n>.json` partition.

Output: `s3://raw-steam-data-*/games/<scrape-date>-<n>-<part>.parquet`, flushed every
`games_flush_every` games; an appid is only marked `scraped` after its row is in S3.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import UTC, date, datetime
from typing import Any

import boto3
import polars as pl

from steam_ingestion.config import IngestionSettings, ScrapeTaskSettings, configure_logging
from steam_ingestion.models import GAMES_PARTITION_RE, GameRecord
from steam_ingestion.schemas import GAMES_SCHEMA
from steam_ingestion.state import GameIdsState
from steam_ingestion.steam_api import SteamApiError, SteamClient
from steam_ingestion.storage import next_part_number, put_parquet, read_partition

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_LANG_FOOTNOTE_RE = re.compile(r"languages with full audio support.*$", re.IGNORECASE | re.DOTALL)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_languages(raw: str | None) -> list[str]:
    """'English<strong>*</strong>, French<br><strong>*</strong>languages…' -> [English, French]"""
    if not raw:
        return []
    text = _LANG_FOOTNOTE_RE.sub("", raw)
    text = html.unescape(_TAG_RE.sub(",", text)).replace("*", "")
    return [lang.strip() for lang in text.split(",") if lang.strip()]


def _requirements(data: dict[str, Any], which: str) -> str | None:
    reqs = data.get("pc_requirements")
    # Steam returns [] instead of {} when there are no requirements.
    return reqs.get(which) if isinstance(reqs, dict) else None


def build_game_record(
    appid: int, data: dict[str, Any], summary: dict[str, Any], scrape_date: date
) -> GameRecord:
    price_overview = data.get("price_overview") or {}
    if "final" in price_overview:
        price = price_overview["final"] / 100
    else:
        price = 0.0 if data.get("is_free") else None
    platforms = data.get("platforms") or {}
    release = data.get("release_date") or {}
    return GameRecord(
        appid=appid,
        name=data.get("name"),
        type=data.get("type"),
        required_age=_int(data.get("required_age")),
        is_free=data.get("is_free"),
        minimum_pc_requirements=_requirements(data, "minimum"),
        recommended_pc_requirements=_requirements(data, "recommended"),
        controller_support=data.get("controller_support"),
        detailed_description=data.get("detailed_description"),
        about_the_game=data.get("about_the_game"),
        short_description=data.get("short_description"),
        supported_languages=parse_languages(data.get("supported_languages")),
        header_image=data.get("header_image"),
        developers=data.get("developers") or [],
        publishers=data.get("publishers") or [],
        price=price,
        categories=[c["description"] for c in data.get("categories") or []],
        genres=[g["description"] for g in data.get("genres") or []],
        windows_support=platforms.get("windows"),
        mac_support=platforms.get("mac"),
        linux_support=platforms.get("linux"),
        release_date=release.get("date") or None,
        coming_soon=release.get("coming_soon"),
        recommendations=_int((data.get("recommendations") or {}).get("total")),
        dlc=[int(d) for d in data.get("dlc") or []],
        review_score=_int(summary.get("review_score")),
        review_score_desc=summary.get("review_score_desc"),
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
    match = GAMES_PARTITION_RE.match(partition_key)
    if not match:
        raise ValueError(f"not a games partition key: {partition_key}")
    today = today or datetime.now(UTC).date()
    prefix = f"games/{today.isoformat()}-{match['n']}-"
    part = next_part_number(s3, settings.raw_bucket, prefix)

    appids = read_partition(s3, settings.partitions_bucket, partition_key).appids
    state = GameIdsState(dynamodb.Table(settings.game_ids_table))
    logger.info("scraping %d games from %s", len(appids), partition_key)

    buffer: list[GameRecord] = []
    failures = unavailable = written = 0

    def flush() -> None:
        nonlocal part, written
        if not buffer:
            return
        df = pl.DataFrame([r.model_dump() for r in buffer], schema=GAMES_SCHEMA)
        key = f"{prefix}{part:04d}.parquet"
        put_parquet(s3, settings.raw_bucket, key, df)
        for record in buffer:
            state.mark_scraped(record.appid, record.recommendations)
        written += len(buffer)
        logger.info("wrote %d games to s3://%s/%s", len(buffer), settings.raw_bucket, key)
        buffer.clear()
        part += 1

    for i, appid in enumerate(appids, 1):
        try:
            data = client.get_app_details(appid)
            if data is None:
                state.mark_unavailable(appid)
                unavailable += 1
            else:
                summary = client.get_review_summary(appid)
                buffer.append(build_game_record(appid, data, summary, today))
        except SteamApiError as exc:
            failures += 1
            status = state.record_failure(appid, settings.max_game_attempts)
            logger.warning("appid %d failed (%s), now %s", appid, exc, status.value)
        if len(buffer) >= settings.games_flush_every:
            flush()
        if i % 500 == 0:
            logger.info("progress %d/%d (failures=%d)", i, len(appids), failures)
    flush()

    logger.info(
        "done: %d written, %d unavailable, %d failed of %d",
        written,
        unavailable,
        failures,
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
    )
    ok = scrape_partition(
        task.partition_key, settings, client, boto3.client("s3"), boto3.resource("dynamodb")
    )
    return 0 if ok else 1
