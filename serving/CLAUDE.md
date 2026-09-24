# serving

Spec: `specs/5-recsys-serving.md`. Human docs: `README.md`.

## Layout

- `src/steam_serving/` (uv project, hatchling, py3.12; deps are numpy, pydantic,
  pydantic-settings and boto3, where boto3 comes from the Lambda runtime and stays out of the
  zip)
  - `config.py`: `ServingSettings`, env > SSM `/serving/*` when `USE_SSM=true`
  - `contracts.py`: stored items (`StoredRecommendations`, `GameDetails`: tolerant readers of
    the inference contracts) and API responses (`RecommendationsResponse`, `ErrorResponse`)
  - `repository.py`: `DynamoRepository`, which does a GetItem, or a BatchGetItem for details
    with retries of unprocessed keys
  - `online.py`: `UserTower` (training's `user_tower.npz` in numpy, `embed`), `OnlineModel`
    (+ the catalog from inference: exact top K, `recommend`, refuses mismatched model ids),
    `BundleLoader` (manifest -> pinned catalog version + the model's user tower, periodic
    refresh, user tower reused while the model is unchanged, keeps the old model on failure)
  - `app.py`: `App.handle(event)`: routing of Function URL events (payload 2.0), validation,
    the popularity fallback, `POST /recommendations` (online), and JSON responses with
    Cache-Control
  - `handler.py`: Lambda entry point `steam_serving.handler.handler` (app cached per container)
- `tests/`: moto DynamoDB seeded with inference-shaped items, plus a small random user tower
  and catalog (`make_user_tower` / `make_catalog` / `publish_bundle`) on versioned moto S3.
  There are no AWS calls.
- `scripts/build_lambda.sh`: the x86_64 manylinux_2_28 zip (numpy 2.3+ has no manylinux2014
  wheels; the python3.12 runtime is AL2023), excluding boto3 and its dependencies.

## Invariants

- Read-only: the Lambda role only has GetItem / BatchGetItem, GetObject(Version) on
  `serving/online/*` and GetObject on `models/*/user_tower.npz`. Inference owns the tables and
  the catalog; training owns the user towers.
- `UserTower.embed` must equal torch's `UserTower.forward` (training/model.py). When the user
  tower changes, change `steam_training.export` (`USER_TOWER_NUMPY_FORMAT`) and this module
  together. When the catalog changes, change `steam_inference.online` / `ONLINE_BUNDLE_FORMAT`
  and `SUPPORTED_MANIFEST_FORMAT`. Parity is tested in
  `inference/tests/test_online_parity.py`, since torch lives there.
- Online ranking is by the score rounded to 4 decimals (as returned), then by lowest appid.
  Raw float order is not reproducible: identical games differ by CPU / BLAS-dependent noise
  (this flipped CI once). Batch uses `torch.topk`.
- The source of truth for the item shape is `inference/src/steam_inference/contracts.py`.
  Stored models ignore unknown fields. When they change, run inference's
  `tests/test_serving_contract.py`, which imports this package.
- `user_id` must be 1-20 digits, so the reserved `__popular__` item is reachable only as the
  fallback or through `/popular`.
- Never leak internals: unhandled errors return a generic 500 and are logged.
- One DynamoDB round trip per table per request. `MAX_LIMIT` stays <= 100 (the BatchGetItem
  limit).
- No heavy dependencies beyond numpy (no torch, no ANN library): the zip is about 19 MB of the
  50 MB direct-upload limit, and a cold start imports numpy and pydantic in about 0.8 s.

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock).

Infra: `infrastructure/serving.tf` (function with 1 GB of memory, Function URL and auth type,
public permissions, client role, SSM, read access to the bundle). The table `game-details`
lives in `inference.tf`. Workflows: `serving-ci.yml`, `serving-cd.yml`. The auth type is the
`serving_auth_type` input of the infrastructure CD.
