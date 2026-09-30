"""DynamoDB scrape state.

`game-ids-state`: one item per appid (PK `appid`, N) holding the game scrape status and whether
the game was unreleased at its last scrape (`coming_soon`, spec 10). The item with `appid = 0`
(not a real app) holds the catalog cursor (`last_modified` of the newest app seen), used as
`if_modified_since` for the next GetAppList call.

`reviews-state-cursor`: one item per appid with `last_review_ts` (newest review creation
timestamp already written to S3), `total_reviews` (used to balance review partitions), and the
backfill of older reviews: `oldest_review_ts` (oldest review written; absent = unknown, a cursor
from before the backfill that was not seeded) and `backfill_complete`, plus `stored_reviews`
(reviews written so far, for the backfill cap; absent = unknown, a cursor from before the counter
that was not seeded) and `backfill_run_id` (the run of the last backfill, so a retried task does
not spend the run's backfill budget twice).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from steam_ingestion.models import GameState, GameStatus

CATALOG_CURSOR_APPID = 0
BATCH_GET_LIMIT = 100


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _scan(table: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
    while True:
        page = table.scan(**kwargs)
        yield from page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


class GameIdsState:
    def __init__(self, table: Any) -> None:
        self._table = table

    def load_all(self) -> dict[int, GameState]:
        states: dict[int, GameState] = {}
        for item in _scan(
            self._table,
            ProjectionExpression="appid, #s, attempts, recommendations, coming_soon",
            ExpressionAttributeNames={"#s": "status"},
        ):
            appid = int(item["appid"])
            if appid == CATALOG_CURSOR_APPID:
                continue
            rec = item.get("recommendations")
            states[appid] = GameState(
                appid=appid,
                status=GameStatus(item["status"]),
                attempts=int(item.get("attempts", 0)),
                recommendations=int(rec) if rec is not None else None,
                coming_soon=bool(item.get("coming_soon", False)),
            )
        return states

    def get_catalog_cursor(self) -> int | None:
        item = self._table.get_item(Key={"appid": CATALOG_CURSOR_APPID}).get("Item")
        return int(item["last_modified"]) if item else None

    def set_catalog_cursor(self, last_modified: int) -> None:
        self._table.put_item(
            Item={
                "appid": CATALOG_CURSOR_APPID,
                "last_modified": last_modified,
                "updated_at": _now(),
            }
        )

    def add_new(self, appids: Iterable[int], run_id: str) -> None:
        now = _now()
        with self._table.batch_writer() as batch:
            for appid in appids:
                batch.put_item(
                    Item={
                        "appid": appid,
                        "status": GameStatus.PENDING.value,
                        "attempts": 0,
                        "first_seen_run": run_id,
                        "first_seen_at": now,
                    }
                )

    def mark_scraped(
        self, appid: int, recommendations: int | None, coming_soon: bool = False
    ) -> None:
        names = {"#s": "status"}
        values: dict[str, Any] = {
            ":s": GameStatus.SCRAPED.value,
            ":t": _now(),
            ":c": coming_soon,
        }
        expr = "SET #s = :s, scraped_at = :t, coming_soon = :c"
        if recommendations is not None:
            expr += ", recommendations = :r"
            values[":r"] = recommendations
        self._table.update_item(
            Key={"appid": appid},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def requeue(self, appids: Iterable[int]) -> int:
        """Back to `pending` (attempts reset) to be scraped again; only `scraped` games.
        Returns how many were re-queued."""
        requeued = 0
        for appid in appids:
            try:
                self._table.update_item(
                    Key={"appid": appid},
                    UpdateExpression="SET #s = :p, attempts = :z, requeued_at = :t",
                    ConditionExpression=Attr("status").eq(GameStatus.SCRAPED.value),
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":p": GameStatus.PENDING.value,
                        ":z": 0,
                        ":t": _now(),
                    },
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                    raise
                continue
            requeued += 1
        return requeued

    def seed_coming_soon(self, appid: int, coming_soon: bool) -> bool:
        """Set `coming_soon` of a game scraped before spec 10. False when the item is missing or
        already has it (written by a scrape), so seeding is idempotent."""
        try:
            self._table.update_item(
                Key={"appid": appid},
                UpdateExpression="SET coming_soon = :c",
                ConditionExpression="attribute_exists(appid) AND attribute_not_exists(coming_soon)",
                ExpressionAttributeValues={":c": coming_soon},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def mark_unavailable(self, appid: int) -> None:
        self._table.update_item(
            Key={"appid": appid},
            UpdateExpression="SET #s = :s, scraped_at = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": GameStatus.UNAVAILABLE.value, ":t": _now()},
        )

    def record_failure(self, appid: int, max_attempts: int) -> GameStatus:
        """Increment attempts; flips to FAILED once `max_attempts` is reached."""
        attrs = self._table.update_item(
            Key={"appid": appid},
            UpdateExpression="ADD attempts :one",
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )["Attributes"]
        if int(attrs["attempts"]) < max_attempts:
            return GameStatus.PENDING
        self._table.update_item(
            Key={"appid": appid},
            UpdateExpression="SET #s = :s",
            ConditionExpression=Attr("status").eq(GameStatus.PENDING.value),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": GameStatus.FAILED.value},
        )
        return GameStatus.FAILED


@dataclass(frozen=True)
class ReviewCursor:
    last_review_ts: int
    total_reviews: int | None
    oldest_review_ts: int | None = None
    backfill_complete: bool = False
    stored_reviews: int | None = None
    backfill_run_id: str | None = None

    @property
    def backfill_pending(self) -> bool:
        """Older reviews may remain. Unknown `oldest_review_ts` (unseeded cursor): not pending."""
        return self.oldest_review_ts is not None and not self.backfill_complete

    def backfill_budget(self, per_run: int, max_stored: int) -> int:
        """Older reviews one run may backfill for this game (0 when none are pending)."""
        if not self.backfill_pending:
            return 0
        return backfill_budget(self.stored_reviews, per_run, max_stored)


def backfill_budget(stored: int | None, per_run: int, max_stored: int) -> int:
    """`per_run` older reviews, down to what is left under `max_stored` stored reviews
    (0 = no cap; an unknown `stored` count is not capped)."""
    if max_stored and stored is not None:
        return max(0, min(per_run, max_stored - stored))
    return per_run


class ReviewsCursorState:
    def __init__(self, table: Any, resource: Any) -> None:
        self._table = table
        self._resource = resource

    def load(self, appids: list[int]) -> dict[int, ReviewCursor]:
        name = self._table.name
        out: dict[int, ReviewCursor] = {}
        for i in range(0, len(appids), BATCH_GET_LIMIT):
            request = {name: {"Keys": [{"appid": a} for a in appids[i : i + BATCH_GET_LIMIT]]}}
            while request:
                resp = self._resource.batch_get_item(RequestItems=request)
                for item in resp["Responses"].get(name, []):
                    out[int(item["appid"])] = _to_cursor(item)
                request = resp.get("UnprocessedKeys") or {}
        return out

    def load_all(self) -> dict[int, ReviewCursor]:
        return {int(item["appid"]): _to_cursor(item) for item in _scan(self._table)}

    def seed_backfill(self, appid: int, oldest_review_ts: int | None, complete: bool) -> bool:
        """Set the backfill fields of an existing cursor that has neither (seeding). False when
        the cursor is missing or already has them, so seeding is idempotent."""
        names = {"#o": "oldest_review_ts", "#c": "backfill_complete"}
        values: dict[str, Any] = {":c": complete, ":t": _now()}
        expr = "SET #c = :c, updated_at = :t"
        if oldest_review_ts is not None:
            expr += ", #o = :o"
            values[":o"] = oldest_review_ts
        try:
            self._table.update_item(
                Key={"appid": appid},
                UpdateExpression=expr,
                ConditionExpression="attribute_exists(appid) AND attribute_not_exists(#o) "
                "AND attribute_not_exists(#c)",
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def seed_stored_reviews(self, appid: int, stored_reviews: int) -> bool:
        """Set `stored_reviews` of an existing cursor that has none (seeding). False when the
        cursor is missing or already has it, so seeding is idempotent."""
        try:
            self._table.update_item(
                Key={"appid": appid},
                UpdateExpression="SET stored_reviews = :s, updated_at = :t",
                ConditionExpression="attribute_exists(appid) "
                "AND attribute_not_exists(stored_reviews)",
                ExpressionAttributeValues={":s": stored_reviews, ":t": _now()},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def save(self, cursors: dict[int, ReviewCursor]) -> None:
        now = _now()
        with self._table.batch_writer() as batch:
            for appid, cursor in cursors.items():
                item: dict[str, Any] = {
                    "appid": appid,
                    "last_review_ts": cursor.last_review_ts,
                    "updated_at": now,
                }
                if cursor.total_reviews is not None:
                    item["total_reviews"] = cursor.total_reviews
                if cursor.oldest_review_ts is not None:
                    item["oldest_review_ts"] = cursor.oldest_review_ts
                if cursor.oldest_review_ts is not None or cursor.backfill_complete:
                    item["backfill_complete"] = cursor.backfill_complete
                if cursor.stored_reviews is not None:
                    item["stored_reviews"] = cursor.stored_reviews
                if cursor.backfill_run_id is not None:
                    item["backfill_run_id"] = cursor.backfill_run_id
                batch.put_item(Item=item)


def _to_cursor(item: dict[str, Any]) -> ReviewCursor:
    total = item.get("total_reviews")
    oldest = item.get("oldest_review_ts")
    stored = item.get("stored_reviews")
    return ReviewCursor(
        last_review_ts=int(item.get("last_review_ts", 0)),
        total_reviews=int(total) if total is not None else None,
        oldest_review_ts=int(oldest) if oldest is not None else None,
        backfill_complete=bool(item.get("backfill_complete", False)),
        stored_reviews=int(stored) if stored is not None else None,
        backfill_run_id=item.get("backfill_run_id"),
    )
