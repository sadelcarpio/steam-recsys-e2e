# Notebooks

Exploratory analysis of the public Steam reviews dataset (spec 16) to decide which features the
two-tower and ranking models should use (spec 17). Local only; nothing here deploys.

| Notebook | What |
|---|---|
| `01_feature_questions` | Spec 17 questions, each tested against a null model and ending in a verdict; findings table at the end |

## Get the data

The notebook reads the files of the Kaggle dataset `sergiodelcarpio/steam-reviews` (~23 GB
unzipped: `reviews-NNN.parquet`, `games.parquet`, `game_tags.parquet`) from `notebooks/data/`
(gitignored). From the repo root:

```bash
uvx --with six kaggle datasets download sergiodelcarpio/steam-reviews -p notebooks/data
unzip -o notebooks/data/steam-reviews.zip -d notebooks/data && rm notebooks/data/steam-reviews.zip
```

`EDA_DATA_DIR` can point anywhere else, including an S3 prefix holding the same files
(`AWS_PROFILE=admin`).

## Run

```bash
cd notebooks
uv sync --group nlp          # nlp: CPU torch + sentence-transformers (embeddings, sentiment)
uv run jupyter lab
```

The first run downloads two small models from Hugging Face (MiniLM, an SST-2 classifier). It reads
every review file a few times for catalog-wide counts and the top review texts, caching the
results in `.cache/`. Delete that folder after downloading a new dataset version. Without the
`nlp` group, the embedding section falls back to TF-IDF and the sentiment section is skipped.

| Env | Default | |
|---|---|---|
| `EDA_DATA_DIR` | `notebooks/data` | folder or `s3://` prefix with the dataset files |
| `EDA_SAMPLE_USERS` | `0.05` | share of users kept (hash of `user_id`, whole histories) |
| `EDA_TEXT_MODELS` | `auto` | `lexical`: TF-IDF only, no sentiment, no downloads |
| `EDA_SEED` | `0` | samples and null models |
| `EDA_CACHE_DIR` | `notebooks/.cache` | |

## Helpers (`steam_eda`)

```python
from steam_eda import load_games, load_reviews, profile_fit, shuffle_games, user_concentration

games = load_games()  # one row per game + tags + parsed release date
reviews = load_reviews(fraction=0.05)  # 5% of users, every column but the review text
```

`signals.py` holds the analysis building blocks: per-user label concentration, leave-one-out
profile fit, lag similarity of a history, null models (`shuffle_games`, `shuffle_order`), binned
means, AUC / Spearman / Gini. `nlp.py` holds the text models with their fallbacks.

## Develop

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

The tests also run the notebook end to end on a small synthetic dataset (`tests/synthetic.py`),
so a broken cell fails CI. The committed notebook keeps the outputs of its last real run: those
outputs are the findings.
