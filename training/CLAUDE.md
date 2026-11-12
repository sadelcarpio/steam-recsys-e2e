# training

Spec: `specs/3-training-pipeline.md`. Human docs: `README.md`.

## Layout

- `src/steam_training/` (uv project, hatchling, py3.12; torch only through the `cpu` / `cu128`
  extras, so always `uv sync --extra cpu` or `--extra cu128`)
  - `config.py`: `TrainingSettings` (jobs, hyperparameters) and `LaunchSettings` (CD launcher),
    env > SSM `/training/*` when `USE_SSM=true`
  - `contracts.py`: Pydantic artifacts contract (`ModelConfig`, `ModelMetadata`,
    `RecallMetrics`, `EvaluationReport`) + S3 layout docstring + `ARCHITECTURE_VERSION`
  - `data.py`: stage 1. `TableSource` (snapshot + streamed batches), `IcebergSource`,
    `load_training_data` (4 streamed passes, keeps only `TrainRows` / `NegativeRows` /
    sampled `EvalRows`), `catalog_at` (streamed latest row per game), `Ragged` (CSR lists)
  - `batching.py`: stage 2. `build_examples` (explicit negatives), `Collator` (history / item-id
    dropout, hard negatives, `reseed(epoch)`), `Batch.to(device)`, `make_loader`
  - `model.py`: `ItemTower`, `UserTower` (frozen `game_table` buffer), `TwoTowerModel`
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
  - `__main__.py`: SageMaker entry (`python -m steam_training {train,promote}`), plus
    `export --model-id` (local backfill of `user_tower.npz`), `track --model-id [--alias]`
    (replay `tracking.json` into MLflow, champion / challenger aliases)
  - `launch.py`: CD launcher (creates and watches the SageMaker job, writes the job summary);
    `METRIC_DEFINITIONS` (log regexes -> CloudWatch curves in the SageMaker console)
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
  `IcebergSource`, `latest_game_rows` / `catalog_from_table` / `attach_game_tags`, `pad_history`,
  `evaluation.embed_catalog`. Keep those APIs stable, or update inference in the same change
  (its CI runs on `training/src/**`).
- Promotion and training run the image of the model's commit (`training:<model_id>`) unless
  `--image-tag` overrides it (GPU image, locally trained models).
- A4 tag profile (`ModelConfig.user_tag_profile`, optional like tags): the user tower holds a
  game -> tags CSR (`game_tag_*` buffers, set from `Catalog.game_tags`, resized on load in
  `UserTower._load_from_state_dict`). `export.supports(model)` is False for it: no
  `user_tower.npz`, and `run_promotion` refuses it until serving's numpy port handles it.
- Tracking (spec 18): `models/<id>/tracking.json` is written before `metadata.json` on every
  run. MLflow is optional (`tracking` extra, imported lazily inside `MlflowSink`); the image
  doesn't install it. Tracking settings are in `NOT_FINGERPRINTED` and out of the metadata
  hyperparameters. A new metric goes in `contracts.RetrievalDiagnostics` (or `EmbeddingHealth`)
  and `tracking.final_metrics`. `EvalRows.user_id` / `activity` are optional (None = unknown).
- Splits: `data.split_window` turns `VALIDATION_FRACTION` / `TEST_FRACTION` / `FIT_ON` into
  (cutoff, end, evaluation). Rows at or after `SplitInfo.end` are never used (only pass 1 sees their timestamps: no training, no
  evaluation, no popularity counts). With `evaluation == "test"` there is no per-epoch
  monitoring. Diagram of every split case: `docs/splits.svg` (keep it in sync).
- Evaluation excludes `EvalRows.reviewed` (every game reviewed before the row) on top of the
  5-game history; scores of models evaluated before that change (stored champion metrics of an
  older architecture) are not comparable with newer ones.

## Commands

`uv sync --extra cpu && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock). torch is routed per extra to the
`pytorch-cpu` / `pytorch-cu128` indexes in `[tool.uv.sources]` (a tool section). Note that
`uv remove torch` deletes that routing, so restore it before re-adding torch.

Infra for this component: `infrastructure/training.tf`. Workflows: `training-ci.yml`,
`training-cd.yml`, `training-promote.yml`.
