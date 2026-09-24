"""DynamoDB scrape state.

`game-ids-state`: one item per appid (PK `appid`, N) holding the game scrape status. The item
with `appid = 0` (not a real app) holds the catalog cursor (`last_modified` of the newest app
seen), used as `if_modified_since` for the next GetAppList call.

`reviews-state-cursor`: one item per appid with `last_review_ts` (newest review creation
timestamp already written to S3) and `total_reviews` (used to balance review partitions).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.conditions import Attr

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
            ProjectionExpression="appid, #s, attempts, recommendations",
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

    def mark_scraped(self, appid: int, recommendations: int | None) -> None:
        names = {"#s": "status"}
        values: dict[str, Any] = {":s": GameStatus.SCRAPED.value, ":t": _now()}
        expr = "SET #s = :s, scraped_at = :t"
        if recommendations is not None:
            expr += ", recommendations = :r"
            values[":r"] = recommendations
        self._table.update_item(
            Key={"appid": appid},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

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

    def load_totals(self) -> dict[int, int]:
        return {
            int(item["appid"]): int(item["total_reviews"])
            for item in _scan(self._table, ProjectionExpression="appid, total_reviews")
            if item.get("total_reviews") is not None
        }

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
                batch.put_item(Item=item)


def _to_cursor(item: dict[str, Any]) -> ReviewCursor:
    total = item.get("total_reviews")
    return ReviewCursor(
        last_review_ts=int(item.get("last_review_ts", 0)),
        total_reviews=int(total) if total is not None else None,
    )
