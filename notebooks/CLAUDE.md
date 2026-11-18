# notebooks

Spec: `specs/17-steam-data-eda.md`. EDA notebooks + the `steam_eda` helper package. Local only:
no AWS writes, no CD, CI only. Human docs: `README.md`.

## Layout

- `src/steam_eda/` (uv project, py3.12, polars; pandas only under seaborn)
  - `config.py`: `EdaSettings` (env `EDA_*` only, no SSM): `data_dir` (local folder or
    `s3://`), `sample_users`, `seed`, `text_models` (`auto` | `lexical`), `cache_dir`
  - `data.py`: `files` / `scan` over the public dataset layout (`reviews*.parquet` or
    `reviews/*.parquet`, `games.parquet`, `game_tags.parquet`), `load_reviews` (hash sample of
    users, text column skipped), `load_games` (+ tags, `parse_release_date`), `cached`
  - `signals.py`: pure helpers (concentration, leave-one-out `profile_fit`, `lag_similarity`,
    null models `shuffle_games` / `shuffle_order`, `binned_mean`, `auc`, `spearman`, `gini`)
  - `nlp.py`: MiniLM embeddings / SST-2 sentiment (optional group `nlp`, CPU torch from the
    explicit `pytorch-cpu` index) with a TF-IDF fallback; `knn`
  - `ranking.py` (spec 9 screen): `GameArrays`, `WeeklyPopularity` (as of the weeks before
    t), `Sequences`, `build_groups` (warm positives after T1, cap per user, train/test at T2),
    `ProxyRetriever` (SVD of positives before T1 + β × recent popularity), `group_features`
    (columns `FEATURES`, families `FAMILIES`), `ranking_metrics`, `group_auc`
- `01_feature_questions.ipynb`: spec 17 questions (Q1–Q8 + extras E1–E4). Each section ends in
  `note(question, evidence, verdict)`, and the last cell tabulates them. Committed **with** the
  outputs of the last real run (they are the findings); regenerate them on the real data,
  never on the synthetic one.
- `02_ranking_features.ipynb`: spec 9 ranker features (R1–R7, same `note` pattern, LightGBM
  `lambdarank`). Committed with the outputs of the last real run, like `01`.
- `tests/`: unit tests + `test_notebook.py`, which executes every `[0-9]*.ipynb` on
  `tests/synthetic.py` (lexical text models, no network)
- `data/`, `.cache/`: gitignored (dataset ~23 GB, cached aggregates / embeddings)

## Invariants

- Never read the `review` text column unless needed (~90% of the bytes); `load_reviews` drops it.
- User samples hash `user_id` (whole histories). Compare signals against a null model, not raw.
- `votes_*`, `playtime_forever`, `num_*` and `game_price` are measured at scrape time: never
  propose them as as-of-t features without saying so.
- Ranking features use only the user's reviews strictly before t and game counts of the weeks
  before t's week (`tests/test_ranking.py` checks that later reviews change nothing).
- Dependencies through `uv add` only (the `nlp` group: `uv add --group nlp ...`).

## Commands

`uv sync && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
