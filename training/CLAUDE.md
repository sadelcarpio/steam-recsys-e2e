# training

Spec: `specs/3-training-pipeline.md` (ranker: `specs/9-ranking-model.md`). Human docs: `README.md`.

## Layout

- `src/steam_training/` (uv project, hatchling, py3.12; torch only through the `cpu` / `cu128`
  dependency groups. `cu128` is a default group, so a plain `uv sync` / `uv run` / `uv add`
  keeps a local GPU venv on CUDA torch; CI and the CPU image use `uv sync --no-group cu128
  --group cpu` and then `uv run --no-sync` (a plain `uv run` would re-sync cu128). Never sync
  a developer's venv with `--group cpu`.)
  - `config.py`: `TrainingSettings` (jobs, hyperparameters) and `LaunchSettings` (CD launcher),
    env > SSM `/training/*` when `USE_SSM=true`
  - `contracts.py`: Pydantic artifacts contract (`ModelConfig`, `ModelMetadata`,
    `RecallMetrics`, `EvaluationReport`) + S3 layout docstring + `ARCHITECTURE_VERSION`
  - `data.py`: stage 1. `TableSource` (snapshot + streamed batches), `IcebergSource`,
    `load_training_data` (4 streamed passes, keeps only `TrainRows` / `NegativeRows` /
    sampled `EvalRows`), `catalog_at` (streamed latest row per game), `Ragged` (CSR lists)
  - `batching.py`: stage 2. `build_examples` (explicit negatives), `Collator` (history / item-id
    dropout, hard negatives, `reseed(epoch)`), `Batch.to(device)`, `make_loader`
  - `model.py`: `ItemTower`, `UserTower` (frozen `game_table` buffer, A3 `HistoryAttention`),
    `TwoTowerModel`
  - `train.py`: stage 3. `retrieval_loss`, `train_model` (sync table → capped mining → epoch →
    checkpoint), `Checkpoints` protocol, `fingerprint`, `resolve_device`
  - `evaluation.py`: stage 4. Pure-torch scoring loops on `device` (`CatalogIndex`, recall@K,
    popularity baseline, `mine_hard_negatives`)
  - `pipeline.py`: `run_training`, `run_promotion` (`FORCE_PROMOTION` overrides a rejection,
    the report keeps the real metrics and reason), `decide`
  - `artifacts.py`: `ArtifactStore` (S3 save / load / promote, `save_user_tower_numpy`),
    `S3Checkpoints`
  - `diagnostics.py`: spec 18 metrics from one scoring pass (`retrieval_diagnostics`: user hit
    rate, activity buckets, coverage, exposure Gini; `embedding_health` collapse checks)
  - `tracking.py`: spec 18 experiment tracking. `RunTracker` (record + optional live MLflow,
    errors never fail training), `MlflowSink` (log, replay, `set_alias` on registered model
    `steam-two-tower`), lineage tags, flat metric names (`final/`, `popularity/`, `lift/`,
    `embeddings/`, `epoch/`, `curves/`)
  - `export.py`: the numpy user tower (`user_tower.npz`, `USER_TOWER_ARRAYS`), which the
    serving Lambda runs without torch
  - `__main__.py`: SageMaker entry (`python -m steam_training {train,promote,rank}`), plus
    `export --model-id` (local backfill of `user_tower.npz`), `track --model-id [--alias]`
    (replay `tracking.json` into MLflow, champion / challenger aliases)
  - `ranker/` (spec 9): `features.py` (`FEATURES` / `FAMILIES`, `FEATURE_SCHEMA_VERSION`,
    `GameTable`, `PopularityTable` / `PopularityCounter`, `compute_features`, pure numpy),
    `data.py` (`load_ranker_data`: groups after a retriever's cutoff via `load_training_data`,
    per-user cap, one extra pass for weekly popularity + later positives, `game_details`),
    `model.py` (`Groups`, `build_groups`, `fit_lambdarank`, `target_positions`,
    `ranking_metrics`: recall / NDCG / MRR), `lists.py` (per-user lists at weekly snapshots:
    `UserReviews`, `snapshot_lists`, `ListAccumulator` precision / recall / MAP),
    `pipeline.py` (`run_ranking`: `python -m steam_training rank`; T2 = the retriever's split
    end when it has one)
  - `pairs.py` (spec 9, phase 3): `python -m steam_training pair` (`PairSettings`):
    `run_pair` scores the serving pair and challenger pairs on groups at or after
    `window_start` (newest retriever split end / cutoff and ranker T2), `decide_pair`
    (retriever gate, then final-list gate), `promote_pair`, `rollback`, `apply_report`
    (`APPLY_REPORT`: promote a `PAIR_DRY_RUN` report's choice if the serving pair is unchanged);
    `serving_pair` bootstraps from `models/champion/` before the first `pair.json`
  - `launch.py`: CD launcher for `train`, `promote`, `rank`, `pair` (creates and watches the
    SageMaker job, writes the job summary); `METRIC_DEFINITIONS` (log regexes -> CloudWatch
    curves in the SageMaker console)
- `tests/`: `conftest.py` builds synthetic marts with learnable taste clusters + `FakeSource`
  (streams small batches). Unit tests, train → promote end to end with moto S3, and a bitwise
  resumed-equals-uninterrupted checkpoint test.
- `mlflow/compose.yaml`: local MLflow server (sqlite + artifacts in a named volume, localhost
  only); image tag = the `mlflow-skinny` version in `uv.lock`.
- `Dockerfile`: image `training` (ECR), `TORCH_VARIANT` build arg; tags `<sha>` + `latest`
  (cpu) or `<sha>-cu128`.

## Invariants (keep them when changing code)

- `IcebergSource.batches` streams through `iceberg_batches`: one data file after another, batch
  by batch within a file (pyiceberg's private `ArrowScan._record_batches_from_scan_tasks_and_deletes`,
  guarded by `tests/test_data.py::test_iceberg_source_*`). Never go back to
  `scan.to_arrow_batch_reader()`: pyiceberg submits every file of the scan at once and reads each
  fully, so the whole mart lands in memory (inference and training share this reader).
- `UserTower.game_table` is a buffer, never a parameter. It is synced from
  `ItemTower.game_embedding` only at the start of each epoch, never after the last epoch (the
  user MLP was trained against that snapshot).
- Ids: 0 = padding, 1 = OOV (`etl` lookups). Ids >= vocab size and unseen games map to OOV
  inside the model (`seen_games` buffer in both towers), so inference needs no mapping code.
- No leakage: explicit negatives and the popularity baseline use the training split only.
  Evaluation catalogs use `game_features` strictly before the cutoff. Promotion evaluates after
  max(candidate cutoff, champion cutoff).
- Memory scales with kept rows: never materialise a whole mart; filter per batch in
  `load_training_data`, and keep every read of one load on the same snapshot.
- Resumability: every random stream in `train_model` is seeded from (seed, epoch); new
  randomness must follow that, or resumed runs stop matching (test in `test_train.py`). Settings
  that do not change training go in `NOT_FINGERPRINTED`.
- Device: models, batches and scoring tensors move to `resolve_device(DEVICE)`; saved state dicts
  are always CPU tensors. Keep numpy in data / collate and torch in the scoring loops.
- Artifact writes: `metadata.json` is written last for a model, and
  `evaluation/champion/metrics.json` last for a promotion. The checkpoint is kept after the
  model is saved (a finished run is extended with more `EPOCHS`; lifecycle expires it in 14 d).
- Tags (spec 8): `VocabSizes.tags` is optional. `None` = a model without the tags module, so
  models trained before tags keep loading. New optional item features follow that pattern;
  anything else that changes the state dict layout needs an `ARCHITECTURE_VERSION` bump.
  `ItemFeatures.tags` is a weighted `Ragged` (weights sum to 1 per row, empty without tags).
- Change `ARCHITECTURE_VERSION` when the state dict layout changes. Update `contracts.py`
  whenever the artifact contract changes.
- `user_tower.npz` mirrors `UserTower.forward` for serving's numpy port
  (`serving/src/steam_serving/online.py`). A change to the user tower must update `export.py`
  (bump `USER_TOWER_NUMPY_FORMAT`) and the numpy port together. Parity with torch is tested in
  `inference/tests/test_online_parity.py`. Promotion of a model without the file deletes the
  champion's copy, so a stale user tower is never paired with a new model.
- Log formats parsed by `launch.METRIC_DEFINITIONS`: the per-epoch line of `train_model` and
  `pipeline.metrics_line`. Change them together (`test_metric_definitions_parse_the_job_logs`).
- `inference/` imports this package (path dependency): `TwoTowerModel`, `ArtifactStore`,
  `IcebergSource`, `latest_game_rows` / `catalog_from_table` / `attach_game_tags`,
  `evaluation.embed_catalog`. Keep those APIs stable, or update inference in the same change
  (its CI runs on `training/src/**`).
- Promotion and training run the image of the model's commit (`training:<model_id>`) unless
  `--image-tag` overrides it (GPU image, locally trained models).
- A4 tag profile (`ModelConfig.user_tag_profile`, optional like tags): the user tower holds a
  game -> tags CSR (`game_tag_*` buffers, set from `Catalog.game_tags`, resized on load in
  `UserTower._load_from_state_dict`). Its `user_tower.npz` is format 2
  (`USER_TOWER_TAG_PROFILE_FORMAT`, + `TAG_PROFILE_ARRAYS`); plain models stay format 1, so a
  serving deploy that only reads format 1 keeps working until an A4 model is promoted. Deploy
  serving (reads 1 and 2) before promoting an A4 model.
- Histories (A3: `ModelConfig.history_length` / `history_encoder`, optional, defaults 5 /
  mean = the old layout) are always derived from `interactions` (the marts no longer store
  one): pass 2 keeps every positive before `end`, each row gets its user's last N positives
  strictly before it, and `TrainRows.history` is a lazy `data.HistoryIndex` (sorted games + a
  pointer per row; `index[rows]` gathers a matrix, `take` stays lazy), never a [rows, N]
  matrix. Cold rows are filtered after that with the draws taken in pass 2, so the kept rows
  are the same as when the mart supplied the warm flag. Reference: the synthetic marts' old
  history column (`test_histories_are_the_last_positives_before_each_row`); on real data the
  derivation matched the mart's 5 games on 99.99% of rows (2026-11-12). The attention encoder (`HistoryAttention`) exports as numpy format 3
  (`USER_TOWER_ATTENTION_FORMAT`, `ATTENTION_ARRAYS`; serving's port, parity in inference's
  `test_online_parity.py`). Inference derives the same history (`steam_inference.features`,
  `Reviews.histories`) at the model's `history_length`. Promotion loads the longer history of
  candidate and champion; `evaluation.model_scores` feeds each model its own prefix (most
  recent first, so the prefix is that model's history).
- Ranker (spec 9): every ranker feature at time t uses only the user's liked games before t
  (at most `RANKER_HISTORY_LENGTH`, what an online request carries) and game counts of the weeks
  before t's week; no feature depends on a retriever version (retrieval enters as rank /
  percentile / normalized gap, never the raw score or tower embeddings). Change a feature's
  definition → bump `FEATURE_SCHEMA_VERSION`. Ranker settings live in `RankerSettings` (never
  in `TrainingSettings`, so they don't touch retriever fingerprints or metadata). A ranker's
  groups come after its retriever's cutoff (the retriever never saw their labels). Naive UTC
  datetimes go through `ranker.data.epoch_seconds` (never `datetime.timestamp()`, which applies
  the local timezone).
- Serving pair (spec 9): `evaluation/champion/pair.json` names the serving (retriever, ranker);
  a pair promotion copies the changed models (`copy_model_to_champion`,
  `copy_ranker_to_champion`, None deletes `rankers/champion/`), appends `history.jsonl`, and
  writes `pair.json` last. `run_promotion` (retriever alone) refuses while a ranker serves.
  Pair comparisons use only groups none of the compared models trained or selected on; a
  ranker must have passed its gate and match `FEATURE_SCHEMA_VERSION`. A rollback steps one
  promotion back (the pair before the current one's last non-rollback promotion).
- `DATA_CACHE_DIR` (local runs): `iceberg_batches` reads through `data.CachingFileIO`, which
  copies each remote data / delete file once (`<dir>/<scheme>/<bucket>/<key>`, `.part` +
  rename) and serves it from disk. Safe because Iceberg files are immutable. It relies on
  pyiceberg opening files via `io.new_input(path).open()` (0.12); guarded by
  `tests/test_data.py::test_caching_file_io_*`.
- Best epoch / early stopping: `train_model` returns `TrainResult` (unpacks as `(model,
  losses)`, plus `best_epoch` / `epochs_trained`). The best weights (CPU copy) go into the
  checkpoint's `best` entry when they aren't its `model`, so a resume keeps them. The early
  stop check runs at the top of each epoch, so a resumed run past the stop point doesn't train.
  Selection uses the monitor rows (a sample of the validation rows the final metrics use, so
  the selected score is slightly optimistic; the test split is the unbiased check).
- `USER_TABLE_SYNC_STEPS`: extra `sync_user_table` calls inside an epoch, always before a
  step (never after the last one), so the user MLP has trained against the table it keeps.
- Tracking (spec 18): `models/<id>/tracking.json` is written before `metadata.json` on every
  run. MLflow is optional (`tracking` extra, imported lazily inside `MlflowSink`); the image
  doesn't install it. Tracking settings are in `NOT_FINGERPRINTED` and out of the metadata
  hyperparameters. A new metric goes in `contracts.RetrievalDiagnostics` (or `EmbeddingHealth`)
  and `tracking.final_metrics`. `EvalRows.user_id` / `activity` are optional (None = unknown).
- Splits: `data.split_window` turns `VALIDATION_FRACTION` / `TEST_FRACTION` / `FIT_ON`, or the
  time-based `HOLDOUT_WEEKS` / `TEST_WEEKS` (counted back from the newest row; spec 9), into
  (cutoff, end, evaluation). Rows at or after `SplitInfo.end` are never used (only pass 1 sees their timestamps: no training, no
  evaluation, no popularity counts). With `evaluation == "test"` there is no per-epoch
  monitoring. Diagram of every split case: `docs/splits.svg` (keep it in sync; the time-based windows are drawn in spec 9 "Time windows"
and not in the svg yet).
- Evaluation excludes `EvalRows.reviewed` (every game reviewed before the row) on top of the
  5-game history; scores of models evaluated before that change (stored champion metrics of an
  older architecture) are not comparable with newer ones.

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .` (GPU torch, the
default group; CPU only: `uv sync --no-group cu128 --group cpu` + `uv run --no-sync ...`)
Add deps only with `uv add` (never edit the lock). torch is routed per dependency group to the
`pytorch-cpu` / `pytorch-cu128` indexes in `[tool.uv.sources]` (a tool section). Note that
`uv remove torch` deletes that routing, so restore it before re-adding torch.

Infra for this component: `infrastructure/training.tf`. Workflows: `training-ci.yml`,
`training-cd.yml`, `training-promote.yml` (modes retriever / pair / rollback),
`training-rank.yml` (ranker job), `training-approve.yml` (the training Step Function's approval),
`training-image-cd.yml` (CPU image on merges, via
`cd.yml`; no training job).
