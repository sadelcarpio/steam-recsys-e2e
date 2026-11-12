"""A tiny dataset in the public layout (spec 16), with planted structure: games belong to tag
clusters, users prefer one cluster, and like on-cluster games more. Used to run the notebook end
to end in CI and to test the loaders."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CLUSTERS = [
    ["Horror", "Survival", "Zombies", "Atmospheric"],
    ["Strategy", "Turn-Based", "4X", "Historical"],
    ["Racing", "Driving", "Simulation", "Automobile"],
    ["Puzzle", "Casual", "Relaxing", "2D"],
    ["RPG", "Open World", "Fantasy", "Story Rich"],
    ["FPS", "Shooter", "Multiplayer", "Competitive"],
]
GENRES = ["Action", "Indie", "Adventure", "Strategy", "Simulation", "RPG"]
WORDS = [
    "dark undead night fear escape",
    "empire army conquer turn map",
    "car track speed race drift",
    "tile block match calm brain",
    "quest dragon hero kingdom magic",
    "gun squad arena match rank",
]
LANGUAGES = ["english", "schinese", "russian", "spanish"]


def write_dataset(root: Path, n_games: int = 240, n_users: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = datetime(2026, 10, 30)
    cluster = rng.integers(0, len(CLUSTERS), n_games)
    release = [snapshot - timedelta(days=int(d)) for d in rng.integers(200, 3000, n_games)]
    popularity = rng.pareto(1.2, n_games) + 1
    price = rng.choice([0.0, 4.99, 9.99, 19.99, 59.99], n_games)
    game_ids = np.arange(10, 10 + n_games) * 10
    coming = rng.random(n_games) < 0.05
    pl.DataFrame(
        {
            "game_id": game_ids,
            "game_name": [f"Game {i}" for i in range(n_games)],
            "game_short_description": [
                f"A {WORDS[c]} game where you {WORDS[c].split()[i % 5]} "
                f"and {WORDS[(c + i) % 6].split()[0]}"
                for i, c in enumerate(cluster)
            ],
            "game_header_image": [None] * n_games,
            "game_release_date": [
                "Coming soon" if cs else r.strftime("%b %-d, %Y" if i % 2 else "%-d %b, %Y")
                for i, (r, cs) in enumerate(zip(release, coming, strict=True))
            ],
            "game_is_free": price == 0,
            "game_price": price,
            "game_developers": [[f"Dev {i % 40}"] for i in range(n_games)],
            "game_publishers": [[f"Pub {i % 15}"] for i in range(n_games)],
            "game_genres": [[GENRES[c], "Indie"] for c in cluster],
            "game_categories": [["Single-player"]] * n_games,
            "game_coming_soon": coming,
        }
    ).write_parquet(root / "games.parquet")
    pl.DataFrame(
        {
            "game_id": game_ids,
            "tag_names": [list(rng.permutation(CLUSTERS[c])) + ["Indie"] for c in cluster],
            "tag_weights": [[500, 400, 300, 200, 100]] * n_games,
            "scraped_at": [snapshot] * n_games,
        }
    ).write_parquet(root / "game_tags.parquet")

    rows: dict[str, list] = {k: [] for k in ["user_id", "game_id", "is_positive", "reviewed_at"]}
    released = np.flatnonzero(~coming)
    for user in range(n_users):
        n = int(min(rng.geometric(0.08), 120))
        taste = rng.integers(0, len(CLUSTERS))
        on = released[cluster[released] == taste]
        weights = popularity[released] * np.where(cluster[released] == taste, 8.0, 1.0)
        picks = rng.choice(
            released, size=min(n, len(released)), replace=False, p=weights / weights.sum()
        )
        start = snapshot - timedelta(days=int(rng.integers(30, 1500)))
        for k, g in enumerate(picks):
            when = max(release[g], start) + timedelta(
                days=k * float(rng.exponential(5)), seconds=int(rng.integers(0, 86_400))
            )
            rows["user_id"].append(user)
            rows["game_id"].append(int(game_ids[g]))
            rows["is_positive"].append(bool(rng.random() < (0.92 if g in on else 0.7)))
            rows["reviewed_at"].append(min(when, snapshot))
    n = len(rows["user_id"])
    reviews = pl.DataFrame(rows).with_columns(
        review_id=pl.int_range(1, n + 1, dtype=pl.Int64),
        user_id=pl.col("user_id").cast(pl.Int64) * 7919 + 13,
        review=pl.when(pl.col("is_positive"))
        .then(pl.lit("great fun loved every minute"))
        .otherwise(pl.lit("boring and broken, refund")),
        language=pl.Series(rng.choice(LANGUAGES, n, p=[0.6, 0.2, 0.1, 0.1])),
        updated_at=pl.col("reviewed_at"),
        playtime_forever=pl.Series(rng.integers(0, 50_000, n)),
        playtime_last_two_weeks=pl.lit(0, dtype=pl.Int64),
        playtime_at_review=pl.Series(rng.integers(0, 20_000, n)),
        num_games_owned=pl.Series(rng.integers(0, 500, n)),
        num_reviews=pl.Series(rng.integers(1, 50, n)),
        votes_up=pl.Series(rng.poisson(1.0, n)),
        votes_funny=pl.Series(np.where(rng.random(n) < 0.01, 4_294_967_295, rng.poisson(0.2, n))),
        weighted_vote_score=pl.Series(rng.random(n)),
        comment_count=pl.lit(0, dtype=pl.Int64),
        steam_purchase=pl.Series(rng.random(n) < 0.8),
        received_for_free=pl.Series(rng.random(n) < 0.05),
        written_during_early_access=pl.Series(rng.random(n) < 0.1),
        primarily_steam_deck=pl.Series(rng.random(n) < 0.02),
    )
    half = reviews.height // 2
    reviews.head(half).write_parquet(root / "reviews-000.parquet")
    reviews.tail(reviews.height - half).write_parquet(root / "reviews-001.parquet")
    return root
