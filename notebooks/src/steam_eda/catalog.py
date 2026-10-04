"""Game-side helpers: id → name lookups and one row per game with everything known about it."""

from __future__ import annotations

from functools import cache

import polars as pl

from steam_eda.sources import iter_mart, load_mart

# list column of a mart -> its lookup mart (ids are dense, 0 = padding, 1 = OOV)
LOOKUPS = {
    "game_developers": "lkp_developers",
    "game_publishers": "lkp_publishers",
    "game_genres": "lkp_genres",
    "game_categories": "lkp_categories",
    "game_tags": "lkp_tags",
}


@cache
def lookup(name: str) -> pl.DataFrame:
    """A lookup mart (`lkp_genres`, …) as `id`, `name`."""
    return load_mart(name, columns=["id", "name"])


def decode(
    df: pl.DataFrame, columns: list[str] | None = None, suffix: str = "_names"
) -> pl.DataFrame:
    """Adds `<column><suffix>` with the names of the ids in each list column (`LOOKUPS`)."""
    columns = columns or [c for c in LOOKUPS if c in df.columns]
    for column in columns:
        names = lookup(LOOKUPS[column])
        mapping = dict(zip(names["id"].to_list(), names["name"].to_list(), strict=True))
        df = df.with_columns(
            pl.col(column)
            .list.eval(pl.element().replace_strict(mapping, default=None, return_dtype=pl.String))
            .alias(f"{column}{suffix}")
        )
    return df


def game_names() -> pl.DataFrame:
    """`game_idx`, `game_id`, `game_name` of every game (latest `game_features` name)."""
    return latest_game_features(columns=["game_idx", "game_id", "game_name"])


def latest_game_features(columns: list[str] | None = None) -> pl.DataFrame:
    """The current row of every game in `game_features` (one row per game and second with
    reviews; the latest is the current state). Reduced while streaming: the mart has 100M+
    rows."""
    keep = list(dict.fromkeys(["game_idx", "timestamp", *(columns or [])])) if columns else None
    latest: pl.DataFrame | None = None
    for batch in iter_mart("game_features", columns=keep):
        if latest is not None:
            batch = pl.concat([latest, batch])
        latest = (
            batch.sort("timestamp")
            .group_by("game_idx", maintain_order=True)
            .last()
            .select(batch.columns)
        )
    if latest is None:
        raise ValueError("game_features is empty")
    latest = latest.sort("game_idx")
    return latest.select(columns) if columns else latest


def review_counts() -> pl.DataFrame:
    """`game_idx`, `reviews`, `positive_reviews` over every interaction (streamed)."""
    parts = [
        batch.group_by("game_idx").agg(
            reviews=pl.len(), positive_reviews=pl.col("is_positive").sum()
        )
        for batch in iter_mart("interactions", columns=["game_idx", "is_positive"])
    ]
    return pl.concat(parts).group_by("game_idx").agg(pl.col("reviews", "positive_reviews").sum())


def game_catalog() -> pl.DataFrame:
    """One row per game: current features (ids), Steam user tags + weights, details (price,
    release date / year, coming soon, short description) and the lifetime review counts."""
    features = latest_game_features(
        ["game_idx", "game_id", "game_name", "game_is_free", "game_developers",
         "game_publishers", "game_genres", "game_categories", "game_reviews_ratio"]
    )  # fmt: skip
    tags = load_mart("game_tags", columns=["game_idx", "game_tags", "game_tag_weights"])
    details = load_mart(
        "game_details",
        columns=["game_id", "game_price", "game_release_date", "game_coming_soon",
                 "game_short_description"],
    )  # fmt: skip
    counts = review_counts()
    return (
        features.join(tags, on="game_idx", how="left")
        .join(details, on="game_id", how="left")
        .join(counts, on="game_idx", how="left")
        .with_columns(
            pl.col("reviews", "positive_reviews").fill_null(0),
            release_year=release_year(pl.col("game_release_date")),
        )
    )


def release_year(text: pl.Expr) -> pl.Expr:
    """Year of Steam's release date text ("21 Aug, 2012", "Aug 21, 2012", "Q3 2026", "2026",
    "Coming soon" → null): the last 4-digit number."""
    return text.str.extract_all(r"\b(?:19|20)\d{2}\b").list.last().cast(pl.Int32, strict=False)
