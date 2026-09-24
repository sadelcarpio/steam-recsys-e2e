# inference

Spec: `specs/4-inference-pipeline.md`. Human docs: `README.md`.

## Layout

- `src/steam_inference/` (uv project, hatchling, py3.12, CPU torch from the `pytorch-cpu`
  index). `steam-training` is an editable path dependency (`../training`): the model, the
  artifact store / contract, `TableSource` / `IcebergSource` and the catalog builders
  (`latest_game_rows`, `catalog_from_table`) come from there. Never copy them.
  - `config.py`: `InferenceSettings`, env > SSM `/inference/*` when `USE_SSM=true`
  - `contracts.py`: DynamoDB item contract (`UserRecommendations.to_item`), the LLM tool output
    (`LlmRanking`), `InferenceSummary`
  - `features.py`: stage 1. Latest `user_features` per user (streamed reduction), latest
    `game_features` per game (all current games, even beyond the model vocab), reviews
    grouped per user and newest first (`Reviews.of` -> CSR), lookup names for prompts
  - `retrieval.py`: stage 2. Exact top K (chunked matmul, reviewed games set to -inf)
  - `rerank.py`: stage 3. Prompt, Bedrock Converse with a forced `submit_ranking` tool,
    `parse_response`, `merge_ranking` (repairs the answer), `rerank_all` (threads, per-user
    failure isolation)
  - `writer.py`: stage 4. `DynamoWriter` (threaded batch writes, a resource per thread),
    `JsonlWriter` (dry runs)
  - `pipeline.py`: `run_inference` (skip when the model is missing), `select_rerank_users`
  - `__main__.py`: ECS entry (`python -m steam_inference`)
- `tests/`: `conftest.py` builds synthetic marts, a random-init model saved as champion
  (moto S3 + DynamoDB) and a fake LLM. There are no AWS calls.
- `Dockerfile`: build context = **repo root** (`docker build -f inference/Dockerfile .`); its
  `Dockerfile.dockerignore` whitelists `training/` + `inference/` sources.

## Invariants

- A missing model (no `models/<MODEL_ID>/metadata.json`) is a successful skip that writes
  nothing. An architecture mismatch fails loudly.
- Never recommend a game the user already reviewed (positive or negative).
- Items are the Pydantic contract (`UserRecommendations`). Serving reads the same shape, so any
  change goes in `contracts.py` + README first. `user_id` is a string, and numbers are `Decimal`.
- LLM output is untrusted: the final order is always a permutation of the retrieved candidates,
  and explanations are kept only for the top `EXPLAIN_TOP_N`. One user's failure never fails
  the run.
- Memory: everything is streamed per batch. Kept arrays are users x K candidates plus every
  review (game id + flag) and one score chunk (`USER_BATCH_SIZE` x games).
- The Bedrock model id lives in Terraform (`inference_bedrock_model_id`), because IAM grants
  exactly that model. Changing it only via env would get AccessDenied in AWS.

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock).

Infra: `infrastructure/inference.tf` (+ the `CheckChampion` / `Infer` states in
`orchestration.tf`). Workflows: `inference-ci.yml`, `inference-cd.yml`.
