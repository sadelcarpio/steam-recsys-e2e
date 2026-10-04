"""Data loading and EDA helpers for the notebooks (see README.md)."""

from steam_eda.cache import cached
from steam_eda.catalog import (
    decode,
    game_catalog,
    game_names,
    latest_game_features,
    lookup,
    review_counts,
)
from steam_eda.features import (
    binned_rate,
    categorical_rate,
    feature_report,
    gini,
    lorenz,
    single_feature_auc,
    tag_profile_similarity,
    with_user_history,
)
from steam_eda.sampling import Sample
from steam_eda.sources import (
    collect,
    iter_mart,
    load_mart,
    load_raw,
    mart_schema,
    mart_table,
    raw_files,
    scan_raw,
    settings,
)

__all__ = [
    "Sample",
    "binned_rate",
    "cached",
    "categorical_rate",
    "collect",
    "decode",
    "feature_report",
    "game_catalog",
    "game_names",
    "iter_mart",
    "gini",
    "latest_game_features",
    "load_mart",
    "load_raw",
    "lookup",
    "lorenz",
    "mart_schema",
    "mart_table",
    "raw_files",
    "review_counts",
    "scan_raw",
    "settings",
    "single_feature_auc",
    "tag_profile_similarity",
    "with_user_history",
]
