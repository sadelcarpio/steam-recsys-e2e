"""Lambda `list-partition-game-ids`: first step of the ingestion state machine.

1. Fetch game appids from GetAppList (only apps modified since the stored catalog cursor).
2. Register unseen appids as `pending` in `game-ids-state`.
3. Write `games/<run_id>/appids-<n>.json`: every pending appid (new + retries), ≤ games_per_task
   per file, one games-scraping task per file.
4. Write `reviews/<run_id>/part-<n>.json`: every known scrapable appid, balanced across
   num_review_workers by estimated request count.

Re-running the same run_id rewrites the same files: new appids are persisted as pending before
the cursor moves, and both prefixes are cleared first.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3

from steam_ingestion.config import IngestionSettings, configure_logging, resolve_steam_api_key
from steam_ingestion.models import (
    GameStatus,
    ListPartitionEvent,
    ListPartitionResult,
    PartitionFile,
)
from steam_ingestion.partitioning import (
    balance_by_weight,
    chunk_by_size,
    estimated_review_requests,
)
from steam_ingestion.state import GameIdsState, ReviewsCursorState
from steam_ingestion.steam_api import SteamClient
from steam_ingestion.storage import delete_prefix, write_partition

logger = logging.getLogger(__name__)


def run(
    run_id: str,
    settings: IngestionSettings,
    client: SteamClient,
    s3: Any,
    dynamodb: Any,
) -> ListPartitionResult:
    games_state = GameIdsState(dynamodb.Table(settings.game_ids_table))
    cursor_state = ReviewsCursorState(dynamodb.Table(settings.reviews_cursor_table), dynamodb)

    known = games_state.load_all()
    catalog_cursor = games_state.get_catalog_cursor()
    apps = client.get_app_list(if_modified_since=catalog_cursor)
    new_ids = sorted({a.appid for a in apps} - known.keys())
    logger.info(
        "catalog: %d known, %d modified since %s, %d new",
        len(known),
        len(apps),
        catalog_cursor,
        len(new_ids),
    )

    # Persist new ids before advancing the cursor so a crash can never lose them.
    games_state.add_new(new_ids, run_id)
    if apps:
        games_state.set_catalog_cursor(max(a.last_modified for a in apps))

    pending = sorted(set(new_ids) | {a for a, s in known.items() if s.status == GameStatus.PENDING})
    games_keys = []
    delete_prefix(s3, settings.partitions_bucket, f"games/{run_id}/")
    for n, chunk in enumerate(chunk_by_size(pending, settings.games_per_task)):
        key = f"games/{run_id}/appids-{n:03d}.json"
        write_partition(
            s3, settings.partitions_bucket, key, PartitionFile(run_id=run_id, appids=chunk)
        )
        games_keys.append(key)

    review_ids = sorted(
        set(new_ids) | {a for a, s in known.items() if s.status != GameStatus.UNAVAILABLE}
    )
    totals = cursor_state.load_totals()
    weights = {
        a: estimated_review_requests(
            totals.get(a, known[a].recommendations if a in known else None),
            settings.max_reviews_per_game,
        )
        for a in review_ids
    }
    reviews_keys = []
    delete_prefix(s3, settings.partitions_bucket, f"reviews/{run_id}/")
    for n, part in enumerate(balance_by_weight(review_ids, weights, settings.num_review_workers)):
        key = f"reviews/{run_id}/part-{n:03d}.json"
        write_partition(
            s3, settings.partitions_bucket, key, PartitionFile(run_id=run_id, appids=part)
        )
        reviews_keys.append(key)

    result = ListPartitionResult(
        run_id=run_id,
        new_game_ids=len(new_ids),
        games_to_scrape=len(pending),
        reviews_game_ids=len(review_ids),
        games_partitions=games_keys,
        reviews_partitions=reviews_keys,
    )
    logger.info("partitions written: %s", result.model_dump_json())
    return result


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    request = ListPartitionEvent.model_validate(event)
    settings = IngestionSettings()
    configure_logging(settings.log_level)
    client = SteamClient(
        resolve_steam_api_key(settings),
        request_interval=0,  # a handful of paginated GetAppList calls; no pacing needed
        max_retries=settings.max_retries,
        max_backoff=settings.max_backoff_seconds,
    )
    result = run(request.run_id, settings, client, boto3.client("s3"), boto3.resource("dynamodb"))
    return result.model_dump()
