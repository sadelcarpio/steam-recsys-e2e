"""Game details for serving: mart `game_details` -> DynamoDB `game-details`, insert-only.

Details are static (the scrape that made the game its name's winner), so only games missing
from the table are written: the first run loads the whole catalog, later runs only new games.
Unreleased games (spec 10) are never written and stored ones are deleted: they are never
recommended, and once released (re-scraped) they are missing again, so their released details
are inserted. A missing mart (ETL not deployed yet) skips the sync.
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
    "game_coming_soon",
]


def read_game_details(
    source: TableSource, snapshot_id: int | None, unreleased: set[int] | None = None
) -> Iterator[GameDetails]:
    """Released games of the mart; the appids of unreleased ones are added to `unreleased`."""
    for batch in source.batches(TABLE, COLUMNS, snapshot_id):
        for row in batch.to_pylist():
            if row["game_id"] is None or not row["game_name"]:
                continue
            if row["game_coming_soon"]:
                if unreleased is not None:
                    unreleased.add(row["game_id"])
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


def sync_game_details(source: TableSource, writer: Writer) -> tuple[int, int, int | None]:
    """Write the released games missing from the table and delete the stored unreleased ones:
    (games written, games deleted, mart snapshot)."""
    try:
        snapshot_id = source.snapshot_id(TABLE)
    except NoSuchTableError:
        log.warning("mart %s not found (ETL not deployed yet?): game details not synced", TABLE)
        return 0, 0, None
    stored = writer.stored_hashes()
    unreleased: set[int] = set()
    written = writer.write(
        g
        for g in read_game_details(source, snapshot_id, unreleased)
        if str(g.game_id) not in stored
    )
    # numeric keys: the table's partition key `game_id` is a number
    deleted = writer.delete(sorted(g for g in unreleased if str(g) in stored))
    log.info(
        "game details: %d new games written, %d unreleased deleted (%d were stored)",
        written,
        deleted,
        len(stored),
    )
    return written, deleted, snapshot_id
