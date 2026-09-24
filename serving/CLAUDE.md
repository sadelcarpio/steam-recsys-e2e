# serving

Spec: `specs/5-recsys-serving.md`. Human docs: `README.md`.

## Layout

- `src/steam_serving/` (uv project, hatchling, py3.12; deps are pydantic, pydantic-settings and
  boto3, where boto3 comes from the Lambda runtime and stays out of the zip)
  - `config.py`: `ServingSettings`, env > SSM `/serving/*` when `USE_SSM=true`
  - `contracts.py`: stored items (`StoredRecommendations`, `GameDetails`: tolerant readers of
    the inference contracts) and API responses (`RecommendationsResponse`, `ErrorResponse`)
  - `repository.py`: `DynamoRepository`, which does a GetItem, or a BatchGetItem for details
    with retries of unprocessed keys
  - `app.py`: `App.handle(event)`: routing of Function URL events (payload 2.0), validation,
    the popularity fallback, and JSON responses with Cache-Control
  - `handler.py`: Lambda entry point `steam_serving.handler.handler` (app cached per container)
- `tests/`: moto DynamoDB seeded with inference-shaped items. There are no AWS calls.
- `scripts/build_lambda.sh`: the x86_64 manylinux zip, excluding boto3 and its dependencies.

## Invariants

- Read-only: the Lambda role only has GetItem / BatchGetItem. Inference owns the tables.
- The source of truth for the item shape is `inference/src/steam_inference/contracts.py`.
  Stored models ignore unknown fields. When they change, run inference's
  `tests/test_serving_contract.py`, which imports this package.
- `user_id` must be 1-20 digits, so the reserved `__popular__` item is reachable only as the
  fallback or through `/popular`.
- Never leak internals: unhandled errors return a generic 500 and are logged.
- One DynamoDB round trip per table per request. `MAX_LIMIT` stays <= 100 (the BatchGetItem
  limit).
- No new heavy dependencies: the zip must stay small (a cold start imports only pydantic).

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock).

Infra: `infrastructure/serving.tf` (function, Function URL and auth type, public permissions,
client role, SSM). The table `game-details` lives in `inference.tf`. Workflows:
`serving-ci.yml`, `serving-cd.yml`. The auth type is the `serving_auth_type` input of the
infrastructure CD.
