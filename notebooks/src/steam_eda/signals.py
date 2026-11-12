"""Pure helpers behind the spec 17 questions (polars in, polars out; tested on small frames).

Conventions: a "reviews" frame has `review_id`, `user_id`, `game_id` (+ `is_positive`,
`reviewed_at`, …); a "labels" frame has `game_id` and one list column of labels (tags, genres,
developers). Every concentration / similarity number is meant to be read against a null model
(`shuffle_games`, `shuffle_order`): the raw value alone mixes taste with popularity.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Row-wise Jaccard of two list columns; null when both are empty or null."""
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(a.list.set_intersection(b).list.len() / union)


def shuffle_games(reviews: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Null model: `game_id` permuted across all rows. Keeps every user's activity and every
    game's popularity, destroys who-reviewed-what (taste)."""
    return reviews.with_columns(pl.col("game_id").shuffle(seed))


def user_concentration(
    reviews: pl.DataFrame, labels: pl.DataFrame, label: str, min_reviews: int = 1
) -> pl.DataFrame:
    """Per user (with ≥ `min_reviews` labelled games): `n_games`, `top_label`, `top_share`
    (share of the user's games carrying their most common label), `entropy` (bits, of the
    label counts) and `effective_labels` (2**entropy)."""
    pairs = (
        reviews.select("user_id", "game_id")
        .unique()
        .join(labels.select("game_id", label), on="game_id")
        .filter(pl.col(label).list.len() > 0)
    )
    sizes = pairs.group_by("user_id").agg(n_games=pl.len()).filter(pl.col("n_games") >= min_reviews)
    counts = (
        pairs.join(sizes, on="user_id")
        .explode(label)
        .group_by("user_id", label)
        .agg(count=pl.len())
    )
    return (
        counts.with_columns(p=pl.col("count") / pl.col("count").sum().over("user_id"))
        .group_by("user_id")
        .agg(
            top_label=pl.col(label).sort_by("count", descending=True).first(),
            top_count=pl.col("count").max(),
            entropy=-(pl.col("p") * pl.col("p").log(2)).sum(),
        )
        .join(sizes, on="user_id")
        .with_columns(
            top_share=pl.col("top_count") / pl.col("n_games"),
            effective_labels=pl.lit(2.0).pow(pl.col("entropy")),
        )
        .select("user_id", "n_games", "top_label", "top_share", "entropy", "effective_labels")
    )


def profile_fit(
    history: pl.DataFrame, scored: pl.DataFrame, labels: pl.DataFrame, label: str
) -> pl.DataFrame:
    """How well each scored game matches its user's history: for each label of the game, the
    share of the user's *other* history games carrying it, averaged over the game's labels
    (0 = nothing in common, 1 = every other history game shares every label).

    `history` and `scored` have `review_id`, `user_id`, `game_id`. A scored row whose
    `review_id` is in `history` is left out of its own profile (leave-one-out), so a review can
    be scored against the history it belongs to. Returns `review_id`, `fit` (null without any
    other history game or labels)."""
    hist = history.select("review_id", "user_id", "game_id").join(
        labels.select("game_id", label), on="game_id"
    )
    n_hist = hist.group_by("user_id").agg(n_hist=pl.len())
    tag_counts = hist.explode(label).group_by("user_id", label).agg(c=pl.len())
    in_hist = hist.select("review_id", self_in=pl.lit(1))
    rows = (
        scored.select("review_id", "user_id", "game_id")
        .join(labels.select("game_id", label), on="game_id")
        .join(in_hist, on="review_id", how="left")
        .join(n_hist, on="user_id", how="left")
        .with_columns(
            pl.col("self_in").fill_null(0),
            pl.col("n_hist").fill_null(0),
        )
        .with_columns(others=pl.col("n_hist") - pl.col("self_in"))
        .filter(pl.col("others") > 0)
        .explode(label)
        .drop_nulls(label)
        .join(tag_counts, on=["user_id", label], how="left")
        .with_columns(share=(pl.col("c").fill_null(0) - pl.col("self_in")) / pl.col("others"))
        .group_by("review_id")
        .agg(fit=pl.col("share").mean())
    )
    return scored.select("review_id").join(rows, on="review_id", how="left")


def sequence(reviews: pl.DataFrame) -> pl.DataFrame:
    """Reviews in each user's order (`reviewed_at`, then `review_id`) with `pos` (0-based)."""
    return reviews.sort("user_id", "reviewed_at", "review_id").with_columns(
        pos=pl.int_range(pl.len()).over("user_id")
    )


def shuffle_order(reviews: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Null model for order: each user's reviews in a random order (same games, same
    `reviewed_at` values, reassigned), so any lag-dependence that survives is an artefact."""
    rng = np.random.default_rng(seed)
    shuffled = reviews.with_columns(_key=pl.Series(rng.random(reviews.height)))
    times = shuffled.sort("user_id", "reviewed_at", "review_id").select("reviewed_at")
    return (
        shuffled.sort("user_id", "_key")
        .drop("reviewed_at", "_key")
        .with_columns(times.to_series())
        .select(reviews.columns)
    )


def lag_similarity(
    reviews: pl.DataFrame, labels: pl.DataFrame, label: str, max_lag: int = 20
) -> pl.DataFrame:
    """Mean Jaccard between the labels of a user's review t and their review t-k, for
    k = 1..`max_lag`, over the same targets for every k (positions ≥ `max_lag`), so each lag
    averages over the same users. A curve that decays with k means order / recency carries
    signal (sequence models, recency-weighted pooling); a flat one means a bag of games is
    enough. Columns: `lag`, `similarity`, `pairs`."""
    seq = sequence(reviews.select("user_id", "game_id", "reviewed_at", "review_id")).join(
        labels.select("game_id", label), on="game_id", how="left"
    )
    seq = seq.with_columns(pl.col(label).fill_null([]))
    out = []
    for lag in range(1, max_lag + 1):
        sim = (
            seq.with_columns(prev=pl.col(label).shift(lag).over("user_id"))
            .filter(pl.col("pos") >= max_lag)
            .select(jaccard(pl.col(label), pl.col("prev")).alias("j"))
            .drop_nulls()
        )
        out.append({"lag": lag, "similarity": sim["j"].mean(), "pairs": sim.height})
    return pl.DataFrame(out, schema={"lag": pl.Int64, "similarity": pl.Float64, "pairs": pl.Int64})


def consecutive_pairs(reviews: pl.DataFrame, labels: pl.DataFrame, label: str) -> pl.DataFrame:
    """Each review with the one before it (same user): `gap_days` and label `similarity`."""
    seq = sequence(reviews.select("user_id", "game_id", "reviewed_at", "review_id")).join(
        labels.select("game_id", label), on="game_id", how="left"
    )
    return (
        seq.with_columns(
            pl.col(label).fill_null([]),
            prev_labels=pl.col(label).fill_null([]).shift(1).over("user_id"),
            prev_time=pl.col("reviewed_at").shift(1).over("user_id"),
        )
        .filter(pl.col("pos") > 0)
        .select(
            "user_id",
            "review_id",
            gap_days=(pl.col("reviewed_at") - pl.col("prev_time")).dt.total_seconds() / 86_400,
            similarity=jaccard(pl.col(label), pl.col("prev_labels")),
        )
    )


def binned_mean(df: pl.DataFrame, x: str, y: str, bins: int = 10) -> pl.DataFrame:
    """Mean of `y` per equal-count bin of `x` (rank based: heavy tails are fine, and tied
    values always share a bin, so a mass at 0 is one bin; nulls dropped). Columns: `x` (bin
    median), `x_min`, `x_max`, `mean`, `rows`."""
    data = df.select(pl.col(x).cast(pl.Float64).alias("x"), pl.col(y).cast(pl.Float64).alias("y"))
    data = data.drop_nulls().drop_nans()
    if data.height == 0:
        return pl.DataFrame(schema={"x": pl.Float64, "x_min": pl.Float64, "x_max": pl.Float64,
                                    "mean": pl.Float64, "rows": pl.UInt32})  # fmt: skip
    return (
        data.with_columns(bin=(pl.col("x").rank("min") - 1) * bins // pl.len())
        .group_by("bin")
        .agg(
            x=pl.col("x").median(),
            x_min=pl.col("x").min(),
            x_max=pl.col("x").max(),
            mean=pl.col("y").mean(),
            rows=pl.len(),
        )
        .sort("bin")
        .drop("bin")
    )


def auc(label: pl.Series | np.ndarray, score: pl.Series | np.ndarray) -> float | None:
    """ROC AUC of one score (nulls / NaN dropped); None when a class is missing."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(label, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(s))
    y, s = y[keep], s[keep]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def gini(counts: pl.Series) -> float:
    """Gini coefficient of non-negative counts (0 = uniform, → 1 = a few take everything)."""
    values = np.sort(counts.drop_nulls().cast(pl.Float64).to_numpy())
    n, total = len(values), values.sum()
    if n == 0 or total == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(((2 * ranks - n - 1) * values).sum() / (n * total))


def smoothed_ratio(
    positives: pl.Expr, total: pl.Expr, prior: float = 0.8, weight: float = 10
) -> pl.Expr:
    """Laplace / Bayesian smoothed positive ratio (`weight` pseudo-reviews at `prior`)."""
    return (positives + prior * weight) / (total + weight)


def spearman(a: pl.Series, b: pl.Series) -> float:
    """Spearman rank correlation (rows with a null on either side dropped)."""
    df = pl.DataFrame({"a": a, "b": b}).drop_nulls().drop_nans()
    if df.height < 3:
        return math.nan
    return float(df.select(pl.corr("a", "b", method="spearman")).item())
