"""Synthetic marts with a learnable structure: games belong to one of N_CLUSTERS genres, each
user likes one cluster (positive reviews inside it, negative reviews outside)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from steam_training.config import TrainingSettings
from steam_training.contracts import USER_HISTORY_LENGTH

N_CLUSTERS = 4
GAMES_PER_CLUSTER = 10
FIRST_GAME = 2
N_GAMES = N_CLUSTERS * GAMES_PER_CLUSTER
START = datetime(2024, 1, 1)
BUCKET = "model-artifacts-test"


@pytest.fixture(autouse=True)
def _no_ssm(monkeypatch) -> None:
    """The CD job sets USE_SSM=true for every step: tests never read Parameter Store."""
    monkeypatch.delenv("USE_SSM", raising=False)
    # nor log to a developer's MLflow server
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")


def game_cluster(game_idx: int) -> int:
    return (game_idx - FIRST_GAME) // GAMES_PER_CLUSTER


def game_id(game_idx):
    """Steam appid of a synthetic game_idx."""
    return game_idx * 10 + 1000


def _item_row(game_idx: int, ratio: float) -> dict:
    cluster = game_cluster(game_idx)
    return {
        "game_idx": game_idx,
        "game_is_free": game_idx % 3 == 0,
        "game_developers": [FIRST_GAME + game_idx % 7],
        "game_publishers": [FIRST_GAME + game_idx % 5],
        "game_genres": [FIRST_GAME + cluster],
        "game_categories": [FIRST_GAME + game_idx % 4, FIRST_GAME + 4],
        "game_reviews_ratio": ratio,
    }


def make_marts(n_users: int = 300, reviews_per_user: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    games = np.arange(FIRST_GAME, FIRST_GAME + N_GAMES)
    events = []
    for user in range(n_users):
        liked = user % N_CLUSTERS
        in_cluster = games[[game_cluster(g) == liked for g in games]]
        out_cluster = games[[game_cluster(g) != liked for g in games]]
        # 3/4 of reviews are positives of the liked cluster
        n_pos = min(reviews_per_user * 3 // 4, len(in_cluster))
        picks = list(rng.choice(in_cluster, n_pos, replace=False))
        negatives = list(rng.choice(out_cluster, reviews_per_user - n_pos, replace=False))
        reviewed = [(g, True) for g in picks] + [(g, False) for g in negatives]
        rng.shuffle(reviewed)
        for game, positive in reviewed:
            ts = START + timedelta(minutes=int(rng.integers(0, 60 * 24 * 365)))
            events.append((ts, user + 10_000, int(game), positive))
    events.sort()

    history: dict[int, list[int]] = {}
    rows = []
    for review_id, (ts, user, game, positive) in enumerate(events):
        past = history.get(user, [])
        padded = past[:USER_HISTORY_LENGTH] + [0] * (USER_HISTORY_LENGTH - len(past))
        rows.append(
            {
                "review_id": review_id,
                "timestamp": ts,
                "user_id": user,
                "is_positive": positive,
                "games_reviewed_positive": padded[:USER_HISTORY_LENGTH],
                **_item_row(game, 0.5 + 0.4 * positive),
            }
        )
        if positive:
            history[user] = [game, *past]

    game_features = [
        {"timestamp": datetime(1970, 1, 1), **_item_row(int(g), 0.5)} for g in games
    ] + [{"timestamp": START + timedelta(days=200), **_item_row(int(g), 0.7)} for g in games]
    # Steam user tags: the cluster's tag (heavy) + a noise tag; the last game has none.
    game_tags = pa.Table.from_pylist(
        [
            {
                "game_idx": int(g),
                "game_tags": [FIRST_GAME + game_cluster(int(g)), FIRST_GAME + 4 + int(g) % 2],
                "game_tag_weights": [900.0, 100.0],
            }
            for g in games[:-1]
        ],
        schema=pa.schema(
            [
                ("game_idx", pa.int64()),
                ("game_tags", pa.list_(pa.int64())),
                ("game_tag_weights", pa.list_(pa.float64())),
            ]
        ),
    )
    # spec 9: reviews per game and ISO week (Monday), as the dbt mart counts them
    weekly: dict[tuple[int, date], list[int]] = {}
    for row in rows:
        monday = row["timestamp"].date() - timedelta(days=row["timestamp"].weekday())
        counts = weekly.setdefault((row["game_idx"], monday), [0, 0])
        counts[0] += int(row["is_positive"])
        counts[1] += 1
    popularity = pa.Table.from_pylist(
        [
            {
                "game_id": game_id(g),
                "game_idx": g,
                "week_start": week,
                "positives": pos,
                "reviews": total,
            }
            for (g, week), (pos, total) in sorted(weekly.items())
        ]
    )
    lookup = lambda n: pa.table({"id": pa.array(range(FIRST_GAME, FIRST_GAME + n), pa.int64())})  # noqa: E731
    return {
        "interactions": pa.Table.from_pylist(rows),
        "game_features": pa.Table.from_pylist(game_features),
        "lkp_games": pa.table(
            {
                "game_idx": pa.array(games, pa.int64()),
                "game_id": pa.array(game_id(games), pa.int64()),
            }
        ),
        # A6 release dates + spec 9 ranker prices
        "game_details": game_details(games),
        "lkp_developers": lookup(7),
        "lkp_publishers": lookup(5),
        "lkp_genres": lookup(N_CLUSTERS),
        "lkp_categories": lookup(5),
        "lkp_tags": lookup(6),
        "game_tags": game_tags,
        "game_popularity_weekly": popularity,
    }


def release_date(game_idx: int) -> str | None:
    """Steam's raw release date strings: mostly exact days, a few unparseable ones."""
    if game_idx % 10 == 7:
        return "Coming soon"
    if game_idx % 10 == 8:
        return "2027"
    if game_idx % 10 == 9:
        return None
    day = START - timedelta(days=30 * (game_idx % 6))
    return f"{day:%b} {day.day}, {day.year}"


def game_details(games) -> pa.Table:
    """A6: release dates (exact, "Coming soon", year only, missing); spec 9 ranker: prices (the
    games without a release date have none either)."""
    games = [int(g) for g in games]
    return pa.table(
        {
            "game_id": pa.array([game_id(g) for g in games], pa.int64()),
            "game_price": pa.array(
                [None if g % 10 == 9 else float(g % 4) * 10 for g in games], pa.float64()
            ),
            "game_release_date": pa.array([release_date(g) for g in games], pa.string()),
        }
    )


class FakeSource:
    """In-memory marts streamed in small batches (exercises the batch-by-batch loader)."""

    def __init__(
        self, tables: dict[str, pa.Table], snapshot: int = 123, batch_rows: int = 257
    ) -> None:
        self.tables = tables
        self.snapshot = snapshot
        self.batch_rows = batch_rows

    def snapshot_id(self, table: str) -> int | None:
        return self.snapshot

    def batches(self, table: str, columns: list[str], snapshot_id: int | None):
        assert snapshot_id == self.snapshot
        yield from self.tables[table].select(columns).to_batches(max_chunksize=self.batch_rows)


@pytest.fixture(scope="session")
def marts() -> dict:
    return make_marts()


@pytest.fixture
def source(marts) -> FakeSource:
    return FakeSource(marts)


@pytest.fixture
def settings(monkeypatch) -> TrainingSettings:
    monkeypatch.delenv("USE_SSM", raising=False)
    return TrainingSettings(
        model_artifacts_bucket=BUCKET,
        model_id="abc123",
        epochs=4,
        batch_size=64,
        learning_rate=5e-3,
        game_embedding_dim=16,
        attribute_embedding_dim=4,
        hidden_dim=32,
        output_dim=16,
        recall_ks=[5, 10],
        primary_k=5,
        epoch_eval_rows=0,
        eval_batch_size=128,
        mine_skip_top=1,
        mine_pool_size=5,
        device="cpu",
    )


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
