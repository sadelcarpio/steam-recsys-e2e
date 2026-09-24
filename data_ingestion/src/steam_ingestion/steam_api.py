"""Thin Steam Web/Store API client with request pacing and retry/backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

APP_LIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
APP_REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}
APP_LIST_PAGE_SIZE = 50_000
REVIEWS_PAGE_SIZE = 100


class SteamApiError(RuntimeError):
    """Raised when retries are exhausted. Never carries the request URL (it holds the key)."""


@dataclass(frozen=True)
class CatalogApp:
    appid: int
    last_modified: int


@dataclass(frozen=True)
class ReviewPage:
    reviews: list[dict[str, Any]]
    # Only present on the first page (cursor="*").
    total_reviews: int | None


class SteamClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_interval: float = 1.5,
        max_retries: int = 5,
        max_backoff: float = 120.0,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._interval = request_interval
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._timeout = timeout
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    # ---- transport ------------------------------------------------------------------------

    def _pace(self) -> None:
        if self._last_request is not None:
            wait = self._interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get_json(self, url: str, params: dict[str, Any], endpoint: str) -> Any:
        """GET with pacing; retries throttling, 5xx, network errors and `null` bodies."""
        for attempt in range(self._max_retries + 1):
            self._pace()
            reason: str
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                reason = type(exc).__name__
            else:
                if resp.status_code in RETRYABLE_STATUS:
                    reason = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise SteamApiError(f"{endpoint}: HTTP {resp.status_code}")
                else:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = None
                    if body is not None:
                        return body
                    reason = "empty body"
            if attempt == self._max_retries:
                break
            backoff = min(self._max_backoff, 2.0 * 2**attempt)
            logger.warning("%s: %s, retry %d in %.0fs", endpoint, reason, attempt + 1, backoff)
            self._sleep(backoff)
        raise SteamApiError(f"{endpoint}: gave up after {self._max_retries + 1} attempts")

    # ---- endpoints ------------------------------------------------------------------------

    def get_app_list(self, if_modified_since: int | None = None) -> list[CatalogApp]:
        """All game appids (no DLC/software/video/hardware), optionally only recently modified."""
        if not self._api_key:
            raise SteamApiError("GetAppList requires an API key")
        params: dict[str, Any] = {
            "key": self._api_key,
            "include_games": "true",
            "include_dlc": "false",
            "include_software": "false",
            "include_videos": "false",
            "include_hardware": "false",
            "max_results": APP_LIST_PAGE_SIZE,
        }
        if if_modified_since:
            params["if_modified_since"] = if_modified_since
        apps: list[CatalogApp] = []
        last_appid = 0
        while True:
            body = self._get_json(APP_LIST_URL, {**params, "last_appid": last_appid}, "GetAppList")
            response = body.get("response") or {}
            apps.extend(
                CatalogApp(appid=int(a["appid"]), last_modified=int(a.get("last_modified", 0)))
                for a in response.get("apps", [])
            )
            if not response.get("have_more_results"):
                return apps
            last_appid = int(response["last_appid"])

    def get_app_details(self, appid: int) -> dict[str, Any] | None:
        """`data` block of appdetails, or None when Steam reports success=false."""
        body = self._get_json(
            APP_DETAILS_URL, {"appids": appid, "cc": "us", "l": "english"}, "appdetails"
        )
        entry = body.get(str(appid)) or {}
        if not entry.get("success"):
            return None
        return entry.get("data")

    def get_review_summary(self, appid: int) -> dict[str, Any]:
        body = self._get_json(
            APP_REVIEWS_URL.format(appid=appid),
            {"json": "1", "language": "all", "purchase_type": "all", "num_per_page": "0"},
            "appreviews",
        )
        return body.get("query_summary") or {}

    def iter_review_pages(
        self, appid: int, since_ts: int = 0, max_reviews: int = 0
    ) -> Iterator[ReviewPage]:
        """Newest-first review pages, stopping at reviews created at/before `since_ts`.

        Yielded pages only contain reviews newer than `since_ts`. `max_reviews` (0 = no cap)
        truncates the walk once that many reviews were yielded.
        """
        cursor = "*"
        seen_cursors: set[str] = set()
        yielded = 0
        while True:
            body = self._get_json(
                APP_REVIEWS_URL.format(appid=appid),
                {
                    "json": "1",
                    "filter": "recent",
                    "language": "all",
                    "purchase_type": "all",
                    "review_type": "all",
                    "cursor": cursor,
                    "num_per_page": str(REVIEWS_PAGE_SIZE),
                },
                "appreviews",
            )
            if not body.get("success", 1):
                return
            total = (
                (body.get("query_summary") or {}).get("total_reviews") if cursor == "*" else None
            )
            raw = body.get("reviews") or []
            fresh = [r for r in raw if int(r.get("timestamp_created", 0)) > since_ts]
            reached_cutoff = len(fresh) < len(raw)
            if max_reviews:
                fresh = fresh[: max_reviews - yielded]
            if fresh or total is not None:
                yield ReviewPage(reviews=fresh, total_reviews=total)
            yielded += len(fresh)
            seen_cursors.add(cursor)
            cursor = body.get("cursor") or ""
            if (
                not raw
                or reached_cutoff
                or (max_reviews and yielded >= max_reviews)
                or not cursor
                or cursor in seen_cursors
            ):
                return
