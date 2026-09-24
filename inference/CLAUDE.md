# inference

Spec: `specs/4-inference-pipeline.md`. Human docs: `README.md`.

## Layout

- `src/steam_inference/` (uv project, hatchling, py3.12, CPU torch from the `pytorch-cpu`
  index). `steam-training` is an editable path dependency (`../training`): the model, the
  artifact store / contract, `TableSource` / `IcebergSource` and the catalog builders
  (`latest_game_rows`, `catalog_from_table`) come from there. Never copy them.
  - `config.py`: `InferenceSettings`, env > SSM `/inference/*` when `USE_SSM=true`
  - `contracts.py`: DynamoDB item contracts (`UserRecommendations.to_item`, `GameDetails`,
    `POPULAR_USER_ID`), the LLM tool output (`LlmRanking`), `InferenceSummary`
  - `features.py`: stage 1. Latest `user_features` per user (streamed reduction), latest
    `game_features` per game (all current games, even beyond the model vocab), reviews
    grouped per user and newest first (`Reviews.of` -> CSR), `popular_counts` (recent positive
    reviews per game_idx), lookup names for prompts
  - `retrieval.py`: stage 2. Exact top K (chunked matmul, reviewed games set to -inf)
  - `rerank.py`: stage 3. Prompt, Bedrock Converse with a forced `submit_ranking` tool,
    `parse_response`, `merge_ranking` (repairs the answer), `rerank_all` (threads, per-user
    failure isolation)
  - `writer.py`: stage 4. `DynamoWriter` (`stored_hashes` = parallel scan, threaded batch
    writes / deletes, a resource per thread), `JsonlWriter` (dry runs, no stored state)
  - `details.py`: mart `game_details` -> `game-details` table, insert-only (`sync_game_details`)
  - `online.py`: the online catalog for serving (`build_catalog`, `encode_catalog`,
    `S3BundleStore` puts the catalog, then a manifest pinned to its version id that names the
    model's `user_tower.npz`; `LocalBundleStore` for dry runs)
  - `pipeline.py`: `run_inference` (skip when the model is missing), `ChangedOnly` (filters
    items whose `content_hash` matches the stored one), `popular_recommendations` (the
    `__popular__` fallback item), `select_rerank_users`
  - `__main__.py`: SageMaker Processing entry point (`python -m steam_inference`)
- `tests/`: `conftest.py` builds synthetic marts, a random-init model saved as champion
  (moto S3 + DynamoDB) and a fake LLM. There are no AWS calls. `test_serving_contract.py`
  reads and serves the written items with `steam_serving` (an editable dev dependency on
  `../serving`; not in the image).
- `Dockerfile`: build context = **repo root** (`docker build -f inference/Dockerfile .`); its
  `Dockerfile.dockerignore` whitelists `training/` + `inference/` sources. Runs as root
  (SageMaker convention).

## Invariants

- A missing model (no `models/<MODEL_ID>/metadata.json`) is a successful skip that writes
  nothing. An architecture mismatch fails loudly.
- Never recommend a game the user already reviewed (positive or negative).
- Only changed items are written: `UserRecommendations.content_hash` covers what the user
  sees, but not the scores or the time. A new visible field must go into the hash, or changes
  to it will never be written. Deletes of users that are gone happen only on full runs
  (`MAX_USERS=0`). There is no TTL, because unchanged items are never rewritten.
- Items are the Pydantic contracts (`UserRecommendations`, `GameDetails`). Serving reads them
  with its own tolerant models (`serving/src/steam_serving/contracts.py`): change both, plus
  the README. The contract test fails when serving can't read an item. `user_id` is a string,
  and numbers are `Decimal`.
- `__popular__` is a reserved `user_id` (never a Steam id). It is always emitted, so it is
  never deleted as a gone user.
- `game-details` is insert-only: an existing game is never rewritten. To force a reload,
  delete the items (or the table) first.
- Online model ownership: training exports the user tower (`steam_training.export`, fixed per
  model), and this pipeline publishes only the catalog side (it depends on the current game
  features). Serving refuses a pair whose `model_id`s differ. The manifest is published only
  when the model has its `user_tower.npz`. `test_online_parity.py` runs training's export and
  this catalog through serving's numpy code, and compares the result with torch and with batch
  retrieval.
- LLM output is untrusted: the final order is always a permutation of the retrieved candidates,
  and explanations are kept only for the top `EXPLAIN_TOP_N`. One user's failure never fails
  the run.
- Memory: everything is streamed per batch. Kept state is users x K candidates, the game id of
  every review, one score chunk (`USER_BATCH_SIZE` x games) and the stored hashes (a dict with
  one entry per item, about 150 MB at 1.4M users).
- The Bedrock model id lives in Terraform (`inference_bedrock_model_id`), because IAM grants
  exactly that model. Changing it only via env would get AccessDenied in AWS.

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock).

Infra: `infrastructure/inference.tf` (table, ECR, SageMaker role) and `orchestration.tf`
(`CheckChampion` / `Infer` states; the job request is `local.inference_job`, reused by the CD). Workflows: `inference-ci.yml`, `inference-cd.yml`.
