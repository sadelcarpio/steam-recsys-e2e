"""Lambda entry point: `steam_serving.handler.handler` (Function URL, payload format 2.0).
Settings and the DynamoDB client are built once per container (cold start); the online bundle
is loaded on the first POST /recommendations and refreshed by `BundleLoader`, the demo user
index manifest on the first GET /users/... (`UserIndex`)."""

from __future__ import annotations

from functools import cache
from typing import Any

from steam_serving.app import App
from steam_serving.config import ServingSettings, configure_logging
from steam_serving.online import BundleLoader
from steam_serving.repository import DynamoRepository
from steam_serving.users import UserIndex


@cache
def app() -> App:
    settings = ServingSettings()
    configure_logging(settings.log_level)
    repository = DynamoRepository(
        settings.recommendations_table, settings.game_details_table, region=settings.aws_region
    )
    online = users = None
    if settings.model_artifacts_bucket:
        online = BundleLoader(
            settings.model_artifacts_bucket,
            settings.online_bundle_prefix,
            region=settings.aws_region,
            refresh_seconds=settings.online_refresh_seconds,
        ).get
        users = UserIndex(
            settings.model_artifacts_bucket,
            settings.user_index_prefix,
            region=settings.aws_region,
            refresh_seconds=settings.user_index_refresh_seconds,
        )
    return App(settings, repository, online, users)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return app().handle(event)
