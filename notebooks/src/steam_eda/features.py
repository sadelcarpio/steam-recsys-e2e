"""Helpers to judge candidate features: leakage-free history features and their signal.

Every history feature of a review at time t uses only the user's reviews strictly before t
(same rule as the marts and the ranker spec), so what the notebooks find carries over to
training. Reviews in the same second are not "before" each other.
"""

from __future__ import annotations

import polars as pl


def with_user_history(
    interactions: pl.DataFrame, user: str = "user_id", time: str = "timestamp"
) -> pl.DataFrame:
    """Adds, per review, the user's counts strictly before it: `prior_reviews`,
    `prior_positives`, `prior_positive_rate` (null without history) and
    `days_since_prior_review` (null for a first review). Needs `is_positive`."""
    per_second = (
        interactions.group_by(user, time)
        .agg(n=pl.len(), pos=pl.col("is_positive").sum().cast(pl.Int64))
        .sort(user, time)
        .with_columns(
            prior_reviews=(pl.col("n").cum_sum() - pl.col("n")).over(user),
            prior_positives=(pl.col("pos").cum_sum() - pl.col("pos")).over(user),
            prior_time=pl.col(time).shift(1).over(user),
        )
    )
    return (
        interactions.join(
            per_second.select(user, time, "prior_reviews", "prior_positives", "prior_time"),
            on=[user, time],
            how="left",
        )
        .with_columns(
            prior_positive_rate=pl.when(pl.col("prior_reviews") > 0).then(
                pl.col("prior_positives") / pl.col("prior_reviews")
            ),
            days_since_prior_review=(pl.col(time) - pl.col("prior_time")).dt.total_seconds()
            / 86400,
        )
        .drop("prior_time")
    )


def binned_rate(
    df: pl.DataFrame, feature: str, target: str = "is_positive", bins: int = 10
) -> pl.DataFrame:
    """Target rate per quantile bin of a numeric feature (nulls get their own bin): a
    monotone or clearly non-flat curve is signal worth a feature. Columns: `bin` (lower edge),
    `rows`, `rate`, `lift` (rate / overall rate)."""
    overall = df[target].cast(pl.Float64).mean()
    binned = df.select(
        pl.col(feature)
        .qcut(bins, allow_duplicates=True, include_breaks=True)
        .struct.field("breakpoint")
        .alias("upper"),
        pl.col(feature).alias("value"),
        pl.col(target).cast(pl.Float64).alias("target"),
    )
    return (
        binned.group_by("upper")
        .agg(
            low=pl.col("value").min(),
            high=pl.col("value").max(),
            rows=pl.len(),
            rate=pl.col("target").mean(),
        )
        .with_columns(lift=pl.col("rate") / overall)
        .sort("low", nulls_last=True)
        .drop("upper")
    )


def categorical_rate(
    df: pl.DataFrame, feature: str, target: str = "is_positive", min_rows: int = 100
) -> pl.DataFrame:
    """Target rate per value of a categorical feature with at least `min_rows` rows, sorted
    by lift. List columns are exploded (one count per element)."""
    overall = df[target].cast(pl.Float64).mean()
    data = df.select(feature, pl.col(target).cast(pl.Float64))
    if isinstance(data.schema[feature], pl.List):
        data = data.explode(feature, empty_as_null=True).drop_nulls(feature)
    return (
        data.group_by(feature)
        .agg(rows=pl.len(), rate=pl.col(target).mean())
        .filter(pl.col("rows") >= min_rows)
        .with_columns(lift=pl.col("rate") / overall)
        .sort("lift", descending=True)
    )


def gini(counts: pl.Series) -> float:
    """Gini coefficient of a non-negative count distribution (0 = uniform, →1 = a few items
    take everything), e.g. reviews per game or per user."""
    values = counts.drop_nulls().cast(pl.Float64).sort()
    n, total = values.len(), values.sum()
    if n == 0 or total == 0:
        return 0.0
    ranks = pl.int_range(1, n + 1, eager=True).cast(pl.Float64)
    return float(((2 * ranks - n - 1) * values).sum() / (n * total))


def lorenz(counts: pl.Series, points: int = 200) -> pl.DataFrame:
    """Share of the total held by the top x% (`top_share` of items vs `share_of_total`), for a
    long-tail plot."""
    values = counts.drop_nulls().cast(pl.Float64).sort(descending=True)
    cumulative = values.cum_sum() / values.sum()
    step = max(1, values.len() // points)
    index = list(range(0, values.len(), step)) + [values.len() - 1]
    return pl.DataFrame(
        {
            "top_share": [(i + 1) / values.len() for i in index],
            "share_of_total": cumulative.gather(index),
        }
    )


def tag_profile_similarity(history_tags: pl.Series, candidate_tags: pl.Series) -> pl.Series:
    """Jaccard overlap of two list columns row by row (e.g. tags of a user's prior positives vs
    the reviewed game's tags): null when either side is empty."""
    df = pl.DataFrame({"a": history_tags, "b": candidate_tags})
    inter = df.select(pl.col("a").list.set_intersection("b").list.len()).to_series()
    union = df.select(pl.col("a").list.set_union("b").list.len()).to_series()
    return pl.select(pl.when(union > 0).then(inter / union)).to_series().alias("jaccard")


def single_feature_auc(df: pl.DataFrame, feature: str, label: str = "label") -> float | None:
    """ROC AUC of one numeric feature on its own (Mann-Whitney, ties averaged; null feature
    rows dropped): 0.5 = no signal, distance from 0.5 = strength (below 0.5: the feature
    separates in the other direction). None when a class is missing."""
    data = df.select(pl.col(feature).cast(pl.Float64), pl.col(label).cast(pl.Boolean)).drop_nulls()
    positives = int(data[label].sum())
    negatives = data.height - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = data.select(pl.col(feature).rank("average"), label)
    rank_sum = ranks.filter(pl.col(label))[feature].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def feature_report(df: pl.DataFrame, features: list[str], label: str = "label") -> pl.DataFrame:
    """`single_feature_auc` and null share of each feature, strongest first."""
    rows = [
        {
            "feature": f,
            "auc": single_feature_auc(df, f, label),
            "null_share": float(df[f].is_null().mean()),
        }
        for f in features
    ]
    return (
        pl.DataFrame(
            rows, schema={"feature": pl.String, "auc": pl.Float64, "null_share": pl.Float64}
        )
        .with_columns(strength=(pl.col("auc") - 0.5).abs())
        .sort("strength", descending=True, nulls_last=True)
    )
