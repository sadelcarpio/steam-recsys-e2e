"""HTTP API of the `recsys-serving` Lambda (Function URL events, payload format 2.0).

    GET /health                              liveness (no DynamoDB call)
    GET /users/{user_id}/recommendations     the user's list, else the popularity fallback
    GET /popular                             the popularity list
    GET /games/{game_id}                     details of one game
    POST /recommendations                    online: for the liked games in the body
                                             {"liked_game_ids": [...], "limit": 10, "details": true}
Query parameters of the GET lists: `limit` (1..MAX_LIMIT, default DEFAULT_LIMIT) and `details`
(true / false: attach each game's details). CORS preflights are answered by the Function URL.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from steam_serving.config import ServingSettings
from steam_serving.contracts import (
    POPULAR_USER_ID,
    USER_ID_PATTERN,
    ErrorResponse,
    GameDetails,
    OnlineRequest,
    RecommendationOut,
    RecommendationsResponse,
    StoredRecommendations,
)
from steam_serving.online import OnlineModel
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
    def __init__(
        self,
        settings: ServingSettings,
        repository: Repository,
        online: Callable[[], OnlineModel | None] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.online = online  # the current online model (None: endpoint disabled)

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        http = event.get("requestContext", {}).get("http", {})
        method = http.get("method", "GET").upper()
        path = event.get("rawPath") or http.get("path") or "/"
        try:
            if method == "POST" and path.rstrip("/") == "/recommendations":
                body, cache = self.online_recommendations(_json_body(event)), False
            elif method != "GET":
                raise HttpError(405, "method not allowed", "GET, or POST /recommendations")
            else:
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

    def online_recommendations(self, payload: Any) -> RecommendationsResponse:
        try:
            request = OnlineRequest.model_validate(payload)
        except ValidationError as err:
            detail = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['msg']}" for e in err.errors()
            )
            raise HttpError(400, "invalid body", detail) from None
        if len(request.liked_game_ids) > self.settings.max_liked_games:
            raise HttpError(
                400, "invalid body", f"at most {self.settings.max_liked_games} liked_game_ids"
            )
        limit, details = self._options(request.limit, request.details)

        model = self.online() if self.online else None
        if model is None:
            raise HttpError(503, "online model not available", "no bundle published yet")
        result = model.recommend(request.liked_game_ids, limit)
        if not result.game_ids:  # no liked game is in the catalog: popular, minus the liked ones
            liked = set(request.liked_game_ids)
            popular = self._popular()
            kept = [r for r in popular.recommendations if r.game_id not in liked]
            response = self._recommendations(
                "popular",
                None,
                popular.model_copy(update={"recommendations": kept}),
                limit,
                details,
            )
        else:
            games = self.repository.games(result.game_ids) if details else {}
            response = RecommendationsResponse(
                source="online",
                user_id=None,
                model_id=model.manifest.model_id,
                generated_at=model.manifest.generated_at,
                reranked=False,
                recommendations=[
                    RecommendationOut(
                        rank=rank,
                        game_id=game_id,
                        name=name,
                        score=score,
                        details=games.get(game_id),
                    )
                    for rank, (game_id, name, score) in enumerate(
                        zip(result.game_ids, result.names, result.scores, strict=True), start=1
                    )
                ],
            )
        return response.model_copy(
            update={
                "used_game_ids": result.used_game_ids,
                "ignored_game_ids": result.ignored_game_ids,
            }
        )

    def game(self, game_id: str) -> GameDetails:
        if not _is_number(game_id) or len(game_id) > 10:
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
        """(limit, details) from the query string of the GET lists."""
        raw_limit, raw_details = query.get("limit"), query.get("details")
        if raw_limit is not None and not _is_number(raw_limit):
            raise HttpError(400, "invalid limit", f"an integer from 1 to {self.settings.max_limit}")
        if raw_details is not None and raw_details.lower() not in _TRUE | _FALSE:
            raise HttpError(400, "invalid details", "true or false")
        return self._options(
            int(raw_limit) if raw_limit is not None else None,
            raw_details.lower() in _TRUE if raw_details is not None else None,
        )

    def _options(self, limit: int | None, details: bool | None) -> tuple[int, bool]:
        """Requested (limit, details), defaults filled in, limit checked."""
        if limit is None:
            limit = self.settings.default_limit
        elif not 1 <= limit <= self.settings.max_limit:
            raise HttpError(400, "invalid limit", f"an integer from 1 to {self.settings.max_limit}")
        return limit, self.settings.include_details if details is None else details

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


def _is_number(value: str) -> bool:
    """ASCII digits only: str.isdigit() also accepts "²" or "٣", which int() rejects or
    reads as other numbers."""
    return value.isascii() and value.isdecimal()


def _json_body(event: dict[str, Any]) -> Any:
    raw = event.get("body") or ""
    try:
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode()
        return json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise HttpError(400, "invalid body", "a JSON object") from None


class _Health(BaseModel):
    status: str
