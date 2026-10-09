## Spec 12: Online multi-stage recommendations (retrieval → ranker → LLM rerank)

### Problem

`POST /recommendations` is a single stage: the numpy user tower, a matmul over the catalog,
top K. Spec 9 adds a LightGBM ranker but only on the batch path, and spec 7's LLM rerank /
explanations only exist in batch. The online path (the one people use from the frontend) gets
neither, and with one stage there is little to observe (spec 11).

Depends on spec 9 (ranker, `ranker.features`) and spec 11 (spans, metrics).

### Behaviour

```
liked games ─► retrieve (user tower, top RETRIEVAL_CANDIDATES=100)
            ─► rank (online ranker, numpy trees)          [when the champion has one]
            ─► rerank / explain (Bedrock, top N)          [only when the request asks]
            ─► top `limit`
```

Each stage is a spec 11 span (`online.retrieve`, `online.features`, `online.rank`,
`online.rerank`, `online.explain`) and the response says which stages ran:
`stages: ["retrieval", "ranker", "llm_rank", "llm_explain"]` (replaces the meaning of
`reranked` for `source = online`; `reranked` stays for compatibility).

**Stage 2: online ranker**

- **A separate model from spec 9's batch ranker.** An online query has only liked games: no
  negative reviews, no timestamps, no review counts. Feeding the batch ranker missing values for
  those would be train / serve skew. The training job trains a second ranker on
  *online-shaped* queries: same time split and groups as spec 9, but the history is the user's
  last `MAX_LIKED_GAMES` positives before t (order kept, no dates) and the features are the
  `ONLINE_FEATURES` subset: retrieval score / rank, candidate, affinity (tags, developer /
  publisher, genres), history similarity, fit, liked count. Same gate (`RANKER_MIN_GAIN` over
  retrieval order on online-shaped test groups); not saved when it fails.
- **No LightGBM in the Lambda** (it pulls scipy: too big for the zip, slow to import).
  Training exports the model as flat arrays, `models/<id>/ranker_online.npz` (per node:
  feature, threshold, left, right, default_left; per leaf: value; trees padded to the same
  node count). Export asserts only numeric `<=` splits and NaN / None missing types (the only
  ones the evaluator implements). Serving evaluates all trees for all candidates at once:
  an index array `[candidates, trees]` walked `max_depth` times with numpy fancy indexing.
- **Candidate data for the features** goes into the online catalog (inference, catalog format
  3, `ONLINE_BUNDLE_ARRAYS` grows): popularity, reviews ratio, price, is_free, release year,
  developer / publisher codes, genre bitmask, top tags as CSR (`tag_ids`, `tag_weights`,
  `tag_offsets`, at most `ONLINE_MAX_TAGS` per game, default 20). The README states the
  catalog size before / after (today ~27 MB; the Lambda has 1 GB).
- **One feature implementation**: `ONLINE_FEATURES` are computed by `steam_serving.ranking`
  (pure numpy over the catalog arrays). Training builds its online-shaped groups with the same
  formulas; a parity test (below) pins them together.
- The manifest names the ranker (`ranker_key`, optional). No ranker (champion without one,
  or `ONLINE_RANKER_ENABLED=false`) → retrieval order, as today.

**Stage 3: LLM rerank / explanations (opt-in per request)**

- Request field `explain: bool` (default false) and setting `ONLINE_LLM_ENABLED` (default
  false). Only when both are true. Ranks stay deterministic without it.
- Reuses spec 7's two calls, with its defaults from `evaluate_rerank` (`RERANK_ENABLED_STAGES`,
  `RERANK_BLEND_WEIGHT`), on the top `ONLINE_RERANK_TOP_N` (default 10) of the ranker:
  stage 1 ranking (permutation check, one retry), stage 2 explanations (grounding checks,
  template fallback). The prompt / tool / check code moves to a small shared module both
  `inference` and `serving` import, or is vendored into `serving` with a test asserting the
  prompts and tool schemas are identical; the implementation picks one and says why.
- **Budget**: `ONLINE_LLM_TIMEOUT_SECONDS` (default 6, Lambda timeout 15) for both calls
  together, no retries past the budget. On timeout, Bedrock error or a failed check: the
  ranker order, template explanations, `llm_fallback_reason` in the request line.
- **Cost guard**: reserved concurrency already caps the Lambda; additionally at most
  `ONLINE_LLM_MAX_CANDIDATES` names per prompt and a per-container token bucket
  (`ONLINE_LLM_RPM`, default 30). Over the limit → no LLM stage, `llm_fallback_reason=rate_limited`.
- Observability (spec 11): spans per call, EMF metrics `LlmLatency`, `LlmInputTokens`,
  `LlmOutputTokens` (from Converse `usage`), `LlmFallbacks` by reason; X-Ray shows the Bedrock
  subsegments (boto3 is patched).
- IAM: `bedrock:InvokeModel` on the configured model only, for the serving role.

**Frontend**

- An "Explain these" toggle on the liked-games view sends `explain: true`; explanations render
  with the existing `.explanation` style. Default off (fast path). A spinner, since the LLM path
  takes seconds.

### Offline evaluation (decides whether stages stay on)

`python -m steam_training rank --online` reports, on online-shaped test groups: recall@{10,30}
and NDCG@30 of retrieval vs online ranker order. The LLM stage on the online path is evaluated
with `evaluate_rerank --online` (candidates from retrieval + online ranker, liked games only).
No online A/B: there is no click signal yet (logging impressions / clicks from the frontend is a
follow-up).

### Out of scope

Online feature store (all features come from the request and the catalog), streaming LLM
output to the browser, caching LLM answers (a DynamoDB cache keyed by the liked set + candidates
is a natural follow-up, and spec 11's metrics will show whether it is worth it), Bedrock
provisioned throughput, using the batch ranker online.

### Tests / docs / infra

- Training: online-shaped group building (history = last positives, no negatives); the export
  to `ranker_online.npz` and its assertions.
- **Parity** (in `inference/tests`, like `test_online_parity.py`): the serving numpy tree
  evaluator matches `lightgbm.Booster.predict` (with NaNs, ties at thresholds) and the serving
  features match the training features for the same history and catalog.
- Serving: each stage on / off, the stages list, LLM timeout / error / failed check → ranker
  order + templates, the rate limit, `explain` without `ONLINE_LLM_ENABLED` is ignored, a
  catalog of format 2 still works (retrieval only) during the rollout. Fake Bedrock client.
- Inference: catalog format 3 arrays; the manifest's `ranker_key`.
- Frontend: the toggle and the explanation rendering (vitest).
- New settings: `ServingSettings` (`ONLINE_RANKER_ENABLED`, `RETRIEVAL_CANDIDATES`,
  `ONLINE_LLM_ENABLED`, `ONLINE_RERANK_TOP_N`, `ONLINE_LLM_TIMEOUT_SECONDS`, `ONLINE_LLM_RPM`,
  `BEDROCK_MODEL_ID`), `InferenceSettings` (`ONLINE_MAX_TAGS`), SSM in
  `infrastructure/serving.tf` / `inference.tf`; Bedrock permission on the serving role.
- Rollout order in `docs/deployment.md`: serving (reads format 2 and 3) → inference (writes
  format 3) → training adds `ranker_online` to the champion (`rank --online --model-id
  champion`) → inference run → `ONLINE_LLM_ENABLED=true` once Bedrock access is back.
- `serving/README.md`, `serving/CLAUDE.md` (invariants: the response is always a ranking of
  catalog games without the liked ones, the LLM stage can only reorder / explain the ranker's
  top N and never blocks past its budget), `training/README.md`, `inference/README.md`,
  `frontend/README.md`, root `CLAUDE.md` (serving online path diagram text).
