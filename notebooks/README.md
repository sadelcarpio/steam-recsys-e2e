# Notebooks

Exploratory analysis of the Steam data: how interactions, users and games are distributed, and which
features carry signal for the towers and the ranker (spec 9). Local only (a workstation with AWS
read access); nothing here deploys.

## Run

```bash
cd notebooks
uv sync
AWS_PROFILE=admin uv run jupyter lab
```

| Notebook | What |
|---|---|
| `00_quickstart` | Loading patterns: marts, samples, lookups, the game catalog, raw parquet, streamed aggregations |
| `01_interactions_eda` | Volume over time + split cutoff, user / game long tails, cold start, history vs sentiment |
| `02_catalog_eda` | Attribute coverage, price / release year / ratio, tags, positive rate by tag / genre |
| `03_feature_signals` | Raw review fields, user history and user × game affinity features scored by single-feature AUC, plus a shortlist template |

Sample sizes are set in each notebook's first cell (env overrides: `EDA_SAMPLE_USERS`,
`EDA_SAMPLE_ROWS`, `EDA_RAW_LAST_FILES`, `EDA_NEGATIVE_SAMPLING`). Notebooks are committed
without outputs.

## Loading data (`steam_eda`)

```python
import polars as pl
from steam_eda import Sample, load_mart, load_raw, game_catalog, cached, decode

# whole histories of 1% of the users, 2025 onwards
df = load_mart(
    "interactions",
    columns=["user_id", "game_idx", "timestamp", "is_positive"],
    where="timestamp >= '2025-01-01T00:00:00'",  # Iceberg pushdown: skips files
    filter=pl.col("is_positive"),  # any polars expression
    sample=Sample.by_key("user_id", 0.01),  # or Sample.rows(0.01)
    limit=None,  # first n rows (fast, biased)
)
catalog = cached("catalog", game_catalog)  # one row per game, cached in .cache/
raw = load_raw("reviews", sample=Sample.by_key("author_id", 0.01))  # review texts skipped
```

| Option | Representative? | Cost |
|---|---|---|
| `where` (time window, ids) | of that slice only | cheap: whole files skipped |
| `Sample.by_key(key, f)` | of users (whole histories, same users across tables) | reads every file, keeps ~f |
| `Sample.rows(f)` | of rows | reads every file, keeps ~f |
| `limit=n` | no (file order) | stops early |

- Marts are streamed one file at a time (`iter_mart` for your own aggregations), so memory
  follows the kept rows. Exception: `game_catalog()` peaks at ~9 GB for 2–3 min, which is why
  it's cached.
- polars throughout; `.to_pandas()` only to plot with seaborn.
- `with_user_history`, `binned_rate`, `categorical_rate`, `feature_report` /
  `single_feature_auc`, `gini`, `lorenz`, `tag_profile_similarity` help screen features.
  History features use only reviews strictly before each review.
- `.cache/` holds full-data results. Refresh with `cached(..., refresh=True)` after a pipeline
  run.

Settings (env): `AWS_REGION`, `MARTS_DATABASE` (`steam_marts`), `RAW_ROOT` (default
`s3://raw-steam-data-<account-id>`, or a local directory), `BATCH_ROWS`, `CACHE_DIR`.

## Develop

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```
