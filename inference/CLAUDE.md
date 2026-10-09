# inference

Specs: `specs/4-inference-pipeline.md`, `specs/7-rerank-two-stage.md` (two-stage rerank +
evaluation), `specs/10-coming-soon-games.md` (unreleased games),
`specs/13-anonymize-user-ids.md` (pseudonymous user ids, demo user index).
Human docs: `README.md`.

## Layout

- `src/steam_inference/` (uv project, hatchling, py3.12, CPU torch from the `pytorch-cpu`
  index). `steam-training` is an editable path dependency (`../training`): the model, the
  artifact store / contract, `TableSource` / `IcebergSource` and the catalog builders
  (`latest_game_rows`, `catalog_from_table`) come from there. Never copy them.
  - `config.py`: `InferenceSettings`, env > SSM `/inference/*` when `USE_SSM=true`
  - `contracts.py`: DynamoDB item contracts (`UserRecommendations.to_item`, `GameDetails`,
    `POPULAR_USER_ID`), the LLM tool inputs (`LlmRanking`, `LlmExplanations`),
    `InferenceSummary`, `RerankEvaluation`
  - `features.py`: stage 1. `review_activity` (pass 1 over `interactions`: reviews /
    positives per user, `Activity.most_active` picks the `MAX_USERS`), `collect_reviews`
    (pass 2: only the kept users' reviews, grouped per user newest first -> `Reviews.of` CSR,
    plus the popularity counts of every user), `latest_users` (latest `user_features` row of
    the kept users; `until` = as-of reads), latest `game_features` per game (all current games,
    even beyond the model vocab), lookup names, `Games.info` / `describe` (prompt lines),
    truncated short descriptions and the release status (`Games.coming_soon`)
  - `retrieval.py`: stage 2. Exact top K (chunked matmul, reviewed games set to -inf)
  - `rerank.py`: stage 3, two calls per user. `llm_ranking` (shuffled prompt,
    `permutation_error` + one retry, else retrieval order), `blend`, then explanations of the
    final top N checked by `explanation_problem` (no candidate name, grounded, length), one
    re-ask of the failed ones, `template_explanation` for the rest. `rerank_user` /
    `rerank_all` (threads, per-user failure isolation, `RerankStats`), `BedrockRerankLlm`
    (forced tools with min/max items, token usage per stage)
  - `evaluate_rerank.py`: `python -m steam_inference.evaluate_rerank`: cohort active after
    the model's cutoff, history as of the cutoff, hit rate@5/@10 of retrieval / LLM / blends
    -> `evaluation/rerank/<model_id>/<ts>.json`
  - `writer.py`: stage 4. `DynamoWriter` (`stored_hashes` = parallel scan, threaded batch
    writes / deletes, a resource per thread), `JsonlWriter` (dry runs, no stored state)
  - `details.py`: mart `game_details` -> `game-details` table, insert-only for released games,
    deletes stored unreleased ones (`sync_game_details`)
  - `online.py`: the online catalog for serving (`build_catalog`, `encode_catalog`,
    `S3BundleStore` puts the catalog, then a manifest pinned to its version id that names the
    model's `user_tower.npz`; `LocalBundleStore` for dry runs; both also publish the search
    index)
  - `search.py`: the frontend's search index (`build_search_index` / `index_from_catalog`:
    catalog games as [appid, name, reviews], `encode_search_index`: gzipped JSON), published
    with the catalog
  - `user_index.py`: `load_user_index` (mart `user_index` -> user ids in `user_idx` order;
    None when missing / empty, raises when not exactly 1..n), `encode_user_index` (int64 LE),
    published by `BundleStore.publish_user_index` (index.bin, then index.json pinned to its
    version; `UserIndexManifest`)
  - `adult.py`: `adult_mask` / `is_adult` (Steam's "Sexual Content" / "Nudity" genres, an
    explicit word in the name, or the adult Steam tags of `ADULT_TAGS` / top `ADULT_TOP_TAGS`)
  - `pipeline.py`: `run_inference` (skip when the model is missing), `ChangedOnly` (filters
    items whose `content_hash` matches the stored one), `popular_recommendations` (the
    `__popular__` fallback item), `select_rerank_users`
  - `__main__.py`: SageMaker Processing entry point (`python -m steam_inference`)
- `scripts/publish_search_index.py`: publishes the search index from the catalog already in
  S3 (+ review counts from one Athena query), without an inference run; drops adult games
  from both (republishing the catalog when it had any)
- `tests/`: `conftest.py` builds synthetic marts, a random-init model saved as champion
  (moto S3 + DynamoDB) and a fake LLM. There are no AWS calls. `test_serving_contract.py`
  reads and serves the written items with `steam_serving` (an editable dev dependency on
  `../serving`; not in the image).
- `Dockerfile`: build context = **repo root** (`docker build -f inference/Dockerfile .`); its
  `Dockerfile.dockerignore` whitelists `training/` + `inference/` sources. Runs as root
  (SageMaker convention).

## Invariants

- Steam tags (`game_tags` / `lkp_tags`, spec 8) are optional marts: loaded when present,
  required when the model's `vocab.tags` is set (it fails loudly otherwise). Deploy this image
  before promoting a model trained with tags: older images reject its metadata.

- A missing model (no `models/<MODEL_ID>/metadata.json`) is a successful skip that writes
  nothing. An architecture mismatch fails loudly.
- Never recommend a game the user already reviewed (positive or negative).
- Adult games (`adult_mask`, on unless `EXCLUDE_ADULT=false`) never reach anything published:
  masked in retrieval (so never candidates, reranked or explained), dropped from the popular
  item, the online catalog (`build_catalog(keep=...)`) and the search index, and left out of
  prompts. A new output must apply the same mask. Unreleased games (`Games.coming_soon`, spec 10)
  are always in the same `excluded` mask.
- Only changed items are written: `UserRecommendations.content_hash` covers what the user
  sees, but not the scores or the time. A new visible field must go into the hash, or changes
  to it will never be written. Deletes of users that are gone happen only on full runs
  (`MAX_USERS=0`) or with `PRUNE_UNSEEN=true` (one-off rekey, spec 13). There is no TTL,
  because unchanged items are never rewritten.
- Items are the Pydantic contracts (`UserRecommendations`, `GameDetails`). Serving reads them
  with its own tolerant models (`serving/src/steam_serving/contracts.py`): change both, plus
  the README. The contract test fails when serving can't read an item. `user_id` is a string,
  and numbers are `Decimal`.
- `user_id` is the pseudonymous id of spec 13 (keyed hash, 63-bit, from the scraper): no
  SteamID64 reaches this pipeline. The demo user index follows the mart `user_index`, whose
  order (num_reviews desc, user_id) must stay the order of `Activity.most_active` and
  `select_rerank_users` (`tests/test_user_index.py`); serving reads it
  (`serving/src/steam_serving/users.py`, `UserIndexManifest`): change both together.
- `__popular__` is a reserved `user_id` (never a real user id). It is always emitted, so it is
  never deleted as a gone user.
- `game-details` is insert-only: an existing game is never rewritten. To force a reload,
  delete the items (or the table) first. It holds released games only: stored unreleased ones
  are deleted, so a release re-inserts the game with its released details.
- Online model ownership: training exports the user tower (`steam_training.export`, fixed per
  model), and this pipeline publishes only the catalog side (it depends on the current game
  features). Serving refuses a pair whose `model_id`s differ. The manifest is published only
  when the model has its `user_tower.npz`. `test_online_parity.py` runs training's export and
  this catalog through serving's numpy code, and compares the result with torch and with batch
  retrieval.
- The search index (`SearchIndex`, `SEARCH_INDEX_FORMAT`) is read by the frontend
  (`frontend/src/api.ts`): change both together. It is published only with the online catalog,
  so search offers exactly the games `POST /recommendations` can use.
- LLM output is untrusted: the final order is always a permutation of the retrieved candidates
  (a ranking is used only when it is an exact permutation). Every top-`EXPLAIN_TOP_N` entry of
  a reranked user has an explanation (checked LLM text or the template), and no other entry
  has one. An explanation never names the recommended game itself. One user's failure never
  fails the run.
- Memory: everything is streamed per batch, and only the `MAX_USERS` kept users' reviews and
  `user_features` rows are held. Kept state is the per-user review counts of pass 1 (3 ints per
  user of `interactions`), the kept users' reviewed game ids, users x K candidates, one score
  chunk (`USER_BATCH_SIZE` x games) and the stored hashes (a dict with one entry per item,
  about 150 MB at 1.4M users). Never load a whole mart when a per-batch filter will do.
- The Bedrock model id lives in Terraform (`inference_bedrock_model_id`), because IAM grants
  exactly that model. Changing it only via env would get AccessDenied in AWS.

## Commands

`uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .`
Add deps only with `uv add` (never edit the lock).

Infra: `infrastructure/inference.tf` (table, ECR, SageMaker role) and `orchestration.tf`
(`CheckChampion` / `Infer` states; the job request is `local.inference_job`, reused by the CD). Workflows: `inference-ci.yml`, `inference-cd.yml`.
