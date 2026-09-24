"""moto DynamoDB tables seeded with items shaped like the inference pipeline's output."""

from __future__ import annotations

import json
from decimal import Decimal

import boto3
import pytest

from steam_serving.app import App
from steam_serving.config import ServingSettings
from steam_serving.repository import DynamoRepository

RECS_TABLE = "game-explainable-recommendations"
DETAILS_TABLE = "game-details"
GENERATED_AT = "2026-09-24T17:00:00+00:00"


def user_item(user_id: str, games: list[int], *, reranked: bool) -> dict:
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "game_id": g,
                "name": f"Game {g}",
                "score": Decimal("0.9") - Decimal(i) / 100,
                **({"explanation": f"why {g}"} if reranked and i < 2 else {}),
            }
            for i, g in enumerate(games)
        ],
        "model_id": "abc123",
        "generated_at": GENERATED_AT,
        "reranked": reranked,
        **({"rerank_model": "us.amazon.nova-2-lite-v1:0"} if reranked else {}),
        "content_hash": "0" * 32,
    }


def popular_item(games: list[int]) -> dict:
    item = user_item("__popular__", games, reranked=False)
    return {**item, "model_id": "popularity"}


def details_item(game_id: int) -> dict:
    item = {
        "game_id": game_id,
        "name": f"Game {game_id}",
        "header_image": f"https://cdn/{game_id}.jpg",
        "is_free": False,
        "price": Decimal("9.99"),
        "developers": ["Studio"],
        "publishers": [],
        "genres": ["Action"],
        "categories": [],
    }
    if game_id % 2:
        item["short_description"] = f"About {game_id}."
    return item


def event(path: str, query: dict | None = None, method: str = "GET") -> dict:
    """Function URL event, payload format 2.0."""
    return {
        "version": "2.0",
        "rawPath": path,
        "queryStringParameters": query,
        "requestContext": {"http": {"method": method, "path": path}},
    }


def body(response: dict) -> dict:
    return json.loads(response["body"])


@pytest.fixture
def aws(monkeypatch):
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("USE_SSM", raising=False)
    with mock_aws():
        client = boto3.client("dynamodb")
        for name, key, key_type in ((RECS_TABLE, "user_id", "S"), (DETAILS_TABLE, "game_id", "N")):
            client.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": key_type}],
                BillingMode="PAY_PER_REQUEST",
            )
        yield


@pytest.fixture
def tables(aws):
    dynamodb = boto3.resource("dynamodb")
    recs, details = dynamodb.Table(RECS_TABLE), dynamodb.Table(DETAILS_TABLE)
    recs.put_item(Item=user_item("76561198000000001", list(range(10, 40)), reranked=True))
    recs.put_item(Item=user_item("76561198000000002", [10, 11, 12], reranked=False))
    recs.put_item(Item=popular_item(list(range(100, 130))))
    for game_id in [*range(10, 40), *range(100, 105)]:  # no details for 105..129
        details.put_item(Item=details_item(game_id))
    return recs, details


@pytest.fixture
def settings() -> ServingSettings:
    return ServingSettings()


@pytest.fixture
def app(tables, settings) -> App:
    return App(settings, DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1"))


# ---- online bundle -----------------------------------------------------------------------------

BUNDLE_BUCKET = "model-artifacts-test"
BUNDLE_PREFIX = "serving/online"
# catalog rows: game_id 10..39 (game_idx 2..31); vocab 30 => game_idx 30, 31 are OOV
ONLINE_GAMES = list(range(10, 40))


USER_TOWER_KEY = "models/abc123/user_tower.npz"


def _npz(arrays: dict) -> bytes:
    import io

    import numpy as np

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def make_user_tower(seed: int = 0, model_id: str = "abc123") -> bytes:
    """models/<model_id>/user_tower.npz as training exports it (steam_training.export)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    vocab, emb, hidden, out = 30, 6, 8, 4
    seen = np.ones(vocab, dtype=bool)
    seen[5] = False  # game_idx 5 (game 13) had no training interactions
    return _npz(
        {
            "format_version": np.int64(1),
            "model_id": np.str_(model_id),
            "history_length": np.int64(5),
            "padding_id": np.int64(0),
            "oov_id": np.int64(1),
            "game_table": rng.normal(size=(vocab, emb)).astype(np.float32),
            "seen_games": seen,
            "w1": rng.normal(size=(hidden, emb + 1)).astype(np.float32),
            "b1": rng.normal(size=hidden).astype(np.float32),
            "w2": rng.normal(size=(out, hidden)).astype(np.float32),
            "b2": rng.normal(size=out).astype(np.float32),
        }
    )


def make_catalog(seed: int = 0) -> bytes:
    """serving/online/catalog.npz as inference writes it (steam_inference.online)."""
    import numpy as np

    rng = np.random.default_rng(seed + 100)
    rows, out = len(ONLINE_GAMES), 4
    items = rng.normal(size=(rows, out)).astype(np.float32)
    names = [f"Game {g} é".encode() for g in ONLINE_GAMES]
    offsets = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum([len(n) for n in names], out=offsets[1:])
    return _npz(
        {
            "item_embeddings": items / np.linalg.norm(items, axis=1, keepdims=True),
            "item_game_id": np.array(ONLINE_GAMES, dtype=np.int64),
            "item_game_idx": np.arange(2, rows + 2, dtype=np.int64),
            "item_name_utf8": np.frombuffer(b"".join(names), dtype=np.uint8),
            "item_name_offsets": offsets,
        }
    )


def make_manifest(catalog: bytes, model_id: str = "abc123", **changes) -> dict:
    import hashlib

    return {
        "format_version": 2,
        "model_id": model_id,
        "generated_at": GENERATED_AT,
        "catalog_key": f"{BUNDLE_PREFIX}/catalog.npz",
        "catalog_version_id": None,
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "catalog_games": len(ONLINE_GAMES),
        "user_tower_key": f"models/{model_id}/user_tower.npz",
        **changes,
    }


def publish_bundle(s3, seed: int = 0, model_id: str = "abc123") -> dict:
    """The model's user tower (training) + a catalog and manifest (inference), on the
    versioned moto bucket."""
    s3.put_object(
        Bucket=BUNDLE_BUCKET,
        Key=f"models/{model_id}/user_tower.npz",
        Body=make_user_tower(seed, model_id),
    )
    catalog = make_catalog(seed)
    put = s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{BUNDLE_PREFIX}/catalog.npz", Body=catalog)
    manifest = make_manifest(catalog, model_id, catalog_version_id=put["VersionId"])
    s3.put_object(
        Bucket=BUNDLE_BUCKET, Key=f"{BUNDLE_PREFIX}/manifest.json", Body=json.dumps(manifest)
    )
    return manifest


@pytest.fixture
def online_model():
    from steam_serving.online import BundleManifest, OnlineModel, UserTower

    catalog = make_catalog()
    manifest = BundleManifest.model_validate(make_manifest(catalog))
    return OnlineModel.from_bytes(manifest, catalog, UserTower.from_bytes(make_user_tower()))


@pytest.fixture
def online_app(tables, settings, online_model) -> App:
    repo = DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1")
    return App(settings, repo, lambda: online_model)
