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
    `load_training_data` (3 streamed passes, keeps only `TrainRows` / `NegativeRows` /
    sampled `EvalRows`), `catalog_at` (streamed latest row per game), `Ragged` (CSR lists)
  - `batching.py`: stage 2. `build_examples` (explicit negatives), `Collator` (history / item-id
    dropout, hard negatives, `reseed(epoch)`), `Batch.to(device)`, `make_loader`
  - `model.py`: `ItemTower`, `UserTower` (frozen `game_table` buffer), `TwoTowerModel`
  - `train.py`: stage 3. `retrieval_loss`, `train_model` (sync table → capped mining → epoch →
    checkpoint), `Checkpoints` protocol, `fingerprint`, `resolve_device`
  - `evaluation.py`: stage 4. Pure-torch scoring loops on `device` (`CatalogIndex`, recall@K,
    popularity baseline, `mine_hard_negatives`)
  - `pipeline.py`: `run_training`, `run_promotion`, `decide`
  - `artifacts.py`: `ArtifactStore` (S3 save / load / promote, `save_user_tower_numpy`),
    `S3Checkpoints`
  - `export.py`: the numpy user tower (`user_tower.npz`, `USER_TOWER_ARRAYS`), which the
    serving Lambda runs without torch
  - `__main__.py`: SageMaker entry (`python -m steam_training {train,promote}`), plus
    `export --model-id` (local backfill of `user_tower.npz`)
  - `launch.py`: CD launcher (creates and watches the SageMaker job, writes the job summary)
- `tests/`: `conftest.py` builds synthetic marts with learnable taste clusters + `FakeSource`
  (streams small batches). Unit tests, train → promote end to end with moto S3, and a bitwise
  resumed-equals-uninterrupted checkpoint test.
- `Dockerfile`: image `training` (ECR), `TORCH_VARIANT` build arg; tags `<sha>` + `latest`
  (cpu) or `<sha>-cu128`.

## Invariants (keep them when changing code)

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
  `evaluation/champion/metrics.json` last for a promotion. The checkpoint is deleted after the
  model is saved.
- Change `ARCHITECTURE_VERSION` when the state dict layout changes. Update `contracts.py`
  whenever the artifact contract changes.
- `user_tower.npz` mirrors `UserTower.forward` for serving's numpy port
  (`serving/src/steam_serving/online.py`). A change to the user tower must update `export.py`
  (bump `USER_TOWER_NUMPY_FORMAT`) and the numpy port together. Parity with torch is tested in
  `inference/tests/test_online_parity.py`. Promotion of a model without the file deletes the
  champion's copy, so a stale user tower is never paired with a new model.
- `inference/` imports this package (path dependency): `TwoTowerModel`, `ArtifactStore`,
  `IcebergSource`, `latest_game_rows` / `catalog_from_table`, `pad_history`,
  `evaluation.embed_catalog`. Keep those APIs stable, or update inference in the same change
  (its CI runs on `training/src/**`).
- Promotion and training run the image of the model's commit (`training:<model_id>`) unless
  `--image-tag` overrides it (GPU image, locally trained models).

## Commands

`uv sync --extra cpu && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock). torch is routed per extra to the
`pytorch-cpu` / `pytorch-cu128` indexes in `[tool.uv.sources]` (a tool section). Note that
`uv remove torch` deletes that routing, so restore it before re-adding torch.

Infra for this component: `infrastructure/training.tf`. Workflows: `training-ci.yml`,
`training-cd.yml`, `training-promote.yml`.
