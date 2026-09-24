"""HTTP API of the `recsys-serving` Lambda (Function URL events, payload format 2.0).

    GET /health                              liveness (no DynamoDB call)
    GET /users/{user_id}/recommendations     the user's list, else the popularity fallback
    GET /popular                             the popularity list
    GET /games/{game_id}                     details of one game
Query parameters of the lists: `limit` (1..MAX_LIMIT, default DEFAULT_LIMIT) and `details`
(true / false: attach each game's details). CORS preflights are answered by the Function URL.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import BaseModel

from steam_serving.config import ServingSettings
from steam_serving.contracts import (
    POPULAR_USER_ID,
    USER_ID_PATTERN,
    ErrorResponse,
    GameDetails,
    RecommendationOut,
    RecommendationsResponse,
    StoredRecommendations,
)
from steam_serving.repository import Repository

log = logging.getLogger(__name__)

USER_RECOMMENDATIONS = re.compile(r"^/users/([^/]+)/recommendations/?$")
GAME = re.compile(r"^/games/([^/]+)/?$")
_USER_ID = re.compile(USER_ID_PATTERN)
_TRUE, _FALSE = {"true", "1", "yes"}, {"false", "0", "no"}


class HttpError(Exception):
    def __init__(self, status: int, error: str, detail: str | None = None) -> None:
        super().__init__(error)
        self.status = status
        self.body = ErrorResponse(error=error, detail=detail)


class App:
    def __init__(self, settings: ServingSettings, repository: Repository) -> None:
        self.settings = settings
        self.repository = repository

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        http = event.get("requestContext", {}).get("http", {})
        method = http.get("method", "GET").upper()
        path = event.get("rawPath") or http.get("path") or "/"
        try:
            if method != "GET":
                raise HttpError(405, "method not allowed", "only GET is supported")
            body, cache = self._route(path, event.get("queryStringParameters") or {})
            status = 200
        except HttpError as err:
            status, body, cache = err.status, err.body, False
        except Exception:
            log.exception("unhandled error on %s %s", method, path)
            status, body, cache = 500, ErrorResponse(error="internal error"), False
        log.info(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "status": status,
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
        )
        return self._response(status, body, cache)

    # ---- routes ----------------------------------------------------------------------------

    def _route(self, path: str, query: dict[str, str]) -> tuple[BaseModel, bool]:
        """(response body, cacheable)."""
        if path in ("/health", "/health/"):
            return _Health(status="ok"), False
        if match := USER_RECOMMENDATIONS.match(path):
            return self.user_recommendations(match.group(1), query), True
        if path in ("/popular", "/popular/"):
            return self.popular(query), True
        if match := GAME.match(path):
            return self.game(match.group(1)), True
        raise HttpError(404, "not found", f"no route for {path}")

    def user_recommendations(self, user_id: str, query: dict[str, str]) -> RecommendationsResponse:
        if not _USER_ID.match(user_id):
            raise HttpError(400, "invalid user_id", "a Steam account id: 1 to 20 digits")
        limit, details = self._list_options(query)
        stored = self.repository.recommendations(user_id)
        if stored is not None:
            return self._recommendations("personalized", user_id, stored, limit, details)
        return self._recommendations("popular", user_id, self._popular(), limit, details)

    def popular(self, query: dict[str, str]) -> RecommendationsResponse:
        limit, details = self._list_options(query)
        return self._recommendations("popular", None, self._popular(), limit, details)

    def game(self, game_id: str) -> GameDetails:
        if not game_id.isdigit() or len(game_id) > 10:
            raise HttpError(400, "invalid game_id", "a Steam appid")
        game = self.repository.game(int(game_id))
        if game is None:
            raise HttpError(404, "game not found")
        return game

    # ---- helpers ---------------------------------------------------------------------------

    def _popular(self) -> StoredRecommendations:
        stored = self.repository.recommendations(POPULAR_USER_ID)
        if stored is None:
            raise HttpError(404, "no recommendations", "no inference run has written them yet")
        return stored

    def _list_options(self, query: dict[str, str]) -> tuple[int, bool]:
        raw_limit = query.get("limit")
        limit = self.settings.default_limit
        if raw_limit is not None:
            if not raw_limit.isdigit() or not 1 <= int(raw_limit) <= self.settings.max_limit:
                raise HttpError(
                    400, "invalid limit", f"an integer from 1 to {self.settings.max_limit}"
                )
            limit = int(raw_limit)
        details = self.settings.include_details
        raw_details = query.get("details")
        if raw_details is not None:
            if raw_details.lower() not in _TRUE | _FALSE:
                raise HttpError(400, "invalid details", "true or false")
            details = raw_details.lower() in _TRUE
        return limit, details

    def _recommendations(
        self,
        source: str,
        user_id: str | None,
        stored: StoredRecommendations,
        limit: int,
        details: bool,
    ) -> RecommendationsResponse:
        recs = stored.recommendations[:limit]
        games = self.repository.games(r.game_id for r in recs) if details and recs else {}
        return RecommendationsResponse(
            source=source,
            user_id=user_id,
            model_id=stored.model_id,
            generated_at=stored.generated_at,
            reranked=stored.reranked,
            rerank_model=stored.rerank_model,
            recommendations=[
                RecommendationOut(
                    rank=rank,
                    game_id=r.game_id,
                    name=r.name,
                    score=r.score,
                    explanation=r.explanation,
                    details=games.get(r.game_id),
                )
                for rank, r in enumerate(recs, start=1)
            ],
        )

    def _response(self, status: int, body: BaseModel, cache: bool) -> dict[str, Any]:
        max_age = self.settings.cache_max_age_seconds if cache and status == 200 else 0
        return {
            "statusCode": status,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": f"public, max-age={max_age}" if max_age else "no-store",
            },
            "body": body.model_dump_json(exclude_none=True),
        }


class _Health(BaseModel):
    status: str
