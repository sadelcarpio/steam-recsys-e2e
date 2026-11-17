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
# Demo user index (spec 13): user_idx -> pseudonymous user_id. Users 1 and 2 have stored
# recommendations, user 3 has none (popular fallback).
USERS = {1: "4559049361332948866", 2: "1234567890123456789", 3: "42"}


def user_item(
    user_id: str, games: list[int], *, reranked: bool, liked: list[int] | None = None
) -> dict:
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
        **({"liked": [{"game_id": g, "name": f"Game {g}"} for g in liked]} if liked else {}),
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
    # liked: 100 has details, 200 has none
    recs.put_item(Item=user_item(USERS[1], list(range(10, 40)), reranked=True, liked=[100, 200]))
    recs.put_item(Item=user_item(USERS[2], [10, 11, 12], reranked=False))
    recs.put_item(Item=popular_item(list(range(100, 130))))
    for game_id in [*range(10, 40), *range(100, 105)]:  # no details for 105..129
        details.put_item(Item=details_item(game_id))
    return recs, details


@pytest.fixture
def settings() -> ServingSettings:
    return ServingSettings()


class FakeUsers:
    """In-memory `steam_serving.users.UserIndex` (its S3 reads are tested in test_users.py)."""

    def __init__(self, users: dict[int, str] | None = None) -> None:
        from steam_serving.users import UserIndexManifest

        self.users = USERS if users is None else users
        self._manifest = (
            UserIndexManifest(
                format_version=1,
                generated_at=GENERATED_AT,
                max_user=len(self.users),
                index_key="serving/users/index.bin",
            )
            if self.users
            else None
        )

    def manifest(self):
        return self._manifest

    def user_id(self, manifest, user_idx: int) -> str:
        return self.users[user_idx]


@pytest.fixture
def app(tables, settings) -> App:
    repo = DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1")
    return App(settings, repo, users=FakeUsers())


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


def publish_user_index(s3, user_ids: list[int], prefix: str = "serving/users") -> dict:
    """serving/users/index.bin + index.json as inference writes them (steam_inference.online)."""
    import hashlib

    import numpy as np

    payload = np.asarray(user_ids, dtype="<i8").tobytes()
    put = s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{prefix}/index.bin", Body=payload)
    manifest = {
        "format_version": 1,
        "generated_at": GENERATED_AT,
        "max_user": len(user_ids),
        "index_key": f"{prefix}/index.bin",
        "index_version_id": put.get("VersionId"),
        "index_sha256": hashlib.sha256(payload).hexdigest(),
    }
    s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{prefix}/index.json", Body=json.dumps(manifest))
    return manifest


FREE_GAMES = [g for g in ONLINE_GAMES if g % 3 == 0]


def make_ranker(
    model_id: str = "abc123", ranker_id: str = "k1", schema: int = 2, candidates: int = 30
) -> bytes:
    """serving/online/ranker.npz as inference writes it (steam_training.ranker.export): one
    tree that puts free games (appid % 3 == 0) first, and the game side of the catalog rows."""
    import numpy as np

    from steam_serving.ranker import FEATURES, POPULARITY_FEATURES

    rows = len(ONLINE_GAMES)
    games = np.array(ONLINE_GAMES)
    tag_offsets = np.arange(rows + 1, dtype=np.int64)  # one tag per game: appid % 4
    return _npz(
        {
            "format_version": np.asarray(1),
            "feature_schema_version": np.asarray(schema),
            "features_utf8": np.frombuffer(",".join(FEATURES).encode(), np.uint8),
            "popularity_features_utf8": np.frombuffer(
                ",".join(POPULARITY_FEATURES).encode(), np.uint8
            ),
            "ranker_id": np.asarray(ranker_id),
            "model_id": np.asarray(model_id),
            "history_length": np.asarray(20),
            "candidates": np.asarray(candidates),
            "as_of_seconds": np.asarray(1_790_000_000),
            # root: is_free <= 0.5 -> leaf 0.0, else leaf 1.0
            "tree_roots": np.array([0], np.int32),
            "node_feature": np.array([FEATURES.index("is_free"), -1, -1], np.int32),
            "node_threshold": np.array([0.5, 0, 0]),
            "node_left": np.array([1, 0, 0], np.int32),
            "node_right": np.array([2, 0, 0], np.int32),
            "node_default_left": np.array([True, False, False]),
            "node_missing": np.array([2, 0, 0], np.int8),
            "node_value": np.array([0.0, 0.0, 1.0]),
            "max_depth": np.asarray(1),
            "game_idx": np.arange(2, rows + 2, dtype=np.int64),
            "game_popularity": np.ones((rows, len(POPULARITY_FEATURES)), np.float32),
            "game_is_free": (games % 3 == 0).astype(np.float32),
            "game_log_price": np.ones(rows, np.float32),
            "game_release_day": np.full(rows, 19_000, np.float32),
            "game_developer": games % 2,
            "game_publisher": np.full(rows, -1),
            "game_tag_values": (games % 4).astype(np.int32),
            "game_tag_weights": np.ones(rows, np.float32),
            "game_tag_offsets": tag_offsets,
            "game_genre_values": np.zeros(0, np.int32),
            "game_genre_offsets": np.zeros(rows + 1, np.int64),
            "game_category_values": np.zeros(0, np.int32),
            "game_category_offsets": np.zeros(rows + 1, np.int64),
            "num_tags": np.asarray(4),
            "num_genres": np.asarray(1),
            "num_categories": np.asarray(1),
        }
    )


def publish_bundle(
    s3,
    seed: int = 0,
    model_id: str = "abc123",
    *,
    ranker: bytes | None = None,
    ranker_mode: str = "on",
    ranker_id: str = "k1",
) -> dict:
    """The model's user tower (training) + a catalog and manifest (inference), on the
    versioned moto bucket; `ranker`: a ranker.npz published with them."""
    import hashlib

    s3.put_object(
        Bucket=BUNDLE_BUCKET,
        Key=f"models/{model_id}/user_tower.npz",
        Body=make_user_tower(seed, model_id),
    )
    fields = {}
    if ranker is not None:
        key = f"{BUNDLE_PREFIX}/ranker.npz"
        put = s3.put_object(Bucket=BUNDLE_BUCKET, Key=key, Body=ranker)
        fields = {
            "ranker_id": ranker_id,
            "ranker_mode": ranker_mode,
            "ranker_key": key,
            "ranker_version_id": put["VersionId"],
            "ranker_sha256": hashlib.sha256(ranker).hexdigest(),
        }
    catalog = make_catalog(seed)
    put = s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{BUNDLE_PREFIX}/catalog.npz", Body=catalog)
    manifest = make_manifest(catalog, model_id, catalog_version_id=put["VersionId"], **fields)
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
