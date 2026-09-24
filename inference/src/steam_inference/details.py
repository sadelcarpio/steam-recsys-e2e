"""Game details for serving: mart `game_details` -> DynamoDB `game-details`, insert-only.

Details are static (the scrape that made the game its name's winner), so only games missing
from the table are written: the first run loads the whole catalog, later runs only new games.
A missing mart (ETL not deployed yet) skips the sync.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from pyiceberg.exceptions import NoSuchTableError
from steam_training.data import TableSource

from steam_inference.contracts import GameDetails, clean_text
from steam_inference.writer import Writer

log = logging.getLogger(__name__)

TABLE = "game_details"
COLUMNS = [
    "game_id",
    "game_name",
    "game_short_description",
    "game_header_image",
    "game_release_date",
    "game_is_free",
    "game_price",
    "game_developers",
    "game_publishers",
    "game_genres",
    "game_categories",
]


def read_game_details(source: TableSource, snapshot_id: int | None) -> Iterator[GameDetails]:
    for batch in source.batches(TABLE, COLUMNS, snapshot_id):
        for row in batch.to_pylist():
            if row["game_id"] is None or not row["game_name"]:
                continue
            price = row["game_price"]
            yield GameDetails(
                game_id=row["game_id"],
                name=row["game_name"],
                short_description=clean_text(row["game_short_description"]),
                header_image=row["game_header_image"] or None,
                release_date=row["game_release_date"] or None,
                is_free=row["game_is_free"],
                price=price if price is not None and price >= 0 else None,
                developers=row["game_developers"] or [],
                publishers=row["game_publishers"] or [],
                genres=row["game_genres"] or [],
                categories=row["game_categories"] or [],
            )


def sync_game_details(source: TableSource, writer: Writer) -> tuple[int, int | None]:
    """Write the games missing from the table: (games written, mart snapshot)."""
    try:
        snapshot_id = source.snapshot_id(TABLE)
    except NoSuchTableError:
        log.warning("mart %s not found (ETL not deployed yet?): game details not synced", TABLE)
        return 0, None
    stored = writer.stored_hashes()
    written = writer.write(
        g for g in read_game_details(source, snapshot_id) if str(g.game_id) not in stored
    )
    log.info("game details: %d new games written (%d already stored)", written, len(stored))
    return written, snapshot_id
