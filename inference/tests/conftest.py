"""Synthetic marts + a small random two-tower model saved as the champion (moto S3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import boto3
import pyarrow as pa
import pytest
import torch
from steam_training.artifacts import ArtifactStore
from steam_training.contracts import (
    ModelConfig,
    ModelMetadata,
    RecallMetrics,
    SegmentRecall,
    SplitInfo,
    VocabSizes,
)
from steam_training.model import TwoTowerModel

from steam_inference.config import InferenceSettings
from steam_inference.contracts import LlmRanking, RankedCandidate

BUCKET = "model-artifacts-test"
TABLE = "game-explainable-recommendations"
FIRST_GAME = 2
N_GAMES = 20  # game_idx 2..21
MODEL_GAMES = FIRST_GAME + N_GAMES - 1  # the model predates the last game (idx 21 -> OOV)
START = datetime(2024, 1, 1)
LIST_TYPE = pa.list_(pa.int64())

# user_id -> reviews (game_idx, is_positive), oldest first
REVIEWS = {
    101: [(2, True), (3, True), (4, False), (5, True), (6, True), (7, True), (8, False)],
    102: [(10, True), (11, True), (12, True), (13, True), (14, True), (15, True)],
    103: [(2, True), (9, False)],
    # reviewed every game but two: fewer candidates than K
    104: [(g, True) for g in range(FIRST_GAME, FIRST_GAME + N_GAMES - 2)],
    # negative reviews only: no user_features row, never recommended
    105: [(3, False)],
}


def game_row(game_idx: int, ts: datetime, ratio: float) -> dict:
    return {
        "game_id": 1000 + game_idx,
        "game_idx": game_idx,
        "timestamp": ts,
        "game_name": f"Game {game_idx}",
        "game_is_free": game_idx % 3 == 0,
        "game_developers": [FIRST_GAME + game_idx % 3],
        "game_publishers": [FIRST_GAME + game_idx % 2],
        "game_genres": [FIRST_GAME + game_idx % 4],
        "game_categories": [FIRST_GAME],
        "game_reviews_ratio": ratio,
    }


def make_marts() -> dict[str, pa.Table]:
    games = range(FIRST_GAME, FIRST_GAME + N_GAMES)
    interactions, user_features = [], []
    for user, reviews in REVIEWS.items():
        liked: list[int] = []
        for i, (game, positive) in enumerate(reviews):
            ts = START + timedelta(days=user - 100, hours=i)
            interactions.append(
                {"user_id": user, "game_idx": game, "is_positive": positive, "timestamp": ts}
            )
            if positive:
                liked = [game, *liked]
                history = (liked + [0] * 5)[:5]
                user_features.append(
                    {"user_id": user, "timestamp": ts, "games_reviewed_positive": history}
                )
    game_features = [game_row(g, datetime(1970, 1, 1), 0.5) for g in games] + [
        game_row(g, START + timedelta(days=30), 0.9) for g in games
    ]
    names = lambda prefix, n: pa.table(  # noqa: E731
        {
            "id": pa.array(range(FIRST_GAME, FIRST_GAME + n), pa.int64()),
            "name": [f"{prefix} {i}" for i in range(n)],
        }
    )
    schema_uf = pa.schema(
        [
            ("user_id", pa.int64()),
            ("timestamp", pa.timestamp("us")),
            ("games_reviewed_positive", LIST_TYPE),
        ]
    )
    return {
        "lkp_games": pa.table({"game_idx": pa.array(list(games), pa.int64())}),
        "lkp_genres": names("Genre", 4),
        "lkp_developers": names("Studio", 3),
        "game_features": pa.Table.from_pylist(game_features),
        "user_features": pa.Table.from_pylist(user_features, schema=schema_uf),
        "interactions": pa.Table.from_pylist(interactions),
    }


class FakeSource:
    """In-memory marts streamed in small batches, pinned to one snapshot."""

    def __init__(self, tables: dict[str, pa.Table], batch_rows: int = 3) -> None:
        self.tables = tables
        self.batch_rows = batch_rows

    def snapshot_id(self, table: str) -> int | None:
        return 7

    def batches(self, table: str, columns: list[str], snapshot_id: int | None):
        assert snapshot_id == 7
        yield from self.tables[table].select(columns).to_batches(max_chunksize=self.batch_rows)


def make_model(seed: int = 0) -> TwoTowerModel:
    torch.manual_seed(seed)
    config = ModelConfig(
        vocab=VocabSizes(games=MODEL_GAMES, developers=5, publishers=4, genres=6, categories=3),
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=16,
        output_dim=8,
        temperature=0.05,
    )
    model = TwoTowerModel(config)
    model.sync_user_table()
    return model.eval()


def make_metadata(model: TwoTowerModel, model_id: str) -> ModelMetadata:
    segment = SegmentRecall(rows=1, recall={5: 0.5})
    metrics = RecallMetrics(warm=segment, cold=segment, all=segment)
    return ModelMetadata(
        model_id=model_id,
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
        config=model.config,
        split=SplitInfo(cutoff=datetime(2024, 5, 1), train_rows=1, validation_rows=1),
        snapshots={},
        hyperparameters={},
        validation=metrics,
        popularity_baseline=metrics,
        epoch_losses=[1.0],
    )


def reverse_ranking(request) -> LlmRanking:
    """Fake LLM: reverses the candidates and explains every one of them."""
    n = len(request.candidates)
    return LlmRanking(
        ranking=[RankedCandidate(candidate=i, explanation=f"because {i}") for i in range(n, 0, -1)]
    )


@pytest.fixture
def marts() -> dict[str, pa.Table]:
    return make_marts()


@pytest.fixture
def source(marts) -> FakeSource:
    return FakeSource(marts)


@pytest.fixture
def settings(monkeypatch) -> InferenceSettings:
    monkeypatch.delenv("USE_SSM", raising=False)
    return InferenceSettings(
        model_artifacts_bucket=BUCKET,
        top_k=5,
        explain_top_n=2,
        rerank_min_reviews=6,
        rerank_max_users=10,
        rerank_concurrency=2,
        user_batch_size=2,
        item_batch_size=7,
        write_concurrency=2,
    )


@pytest.fixture
def aws(monkeypatch):
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket=BUCKET)
        boto3.client("dynamodb").create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield


@pytest.fixture
def store(aws) -> ArtifactStore:
    return ArtifactStore(BUCKET)


@pytest.fixture
def champion(store) -> TwoTowerModel:
    """A trained model promoted to models/champion/ (saved under both ids)."""
    model = make_model()
    store.save_model(model, make_metadata(model, "abc123"))
    for name in ("user_tower.pt", "item_tower.pt", "metadata.json"):
        store.s3.copy_object(
            Bucket=BUCKET,
            Key=f"models/champion/{name}",
            CopySource={"Bucket": BUCKET, "Key": f"models/abc123/{name}"},
        )
    return model
