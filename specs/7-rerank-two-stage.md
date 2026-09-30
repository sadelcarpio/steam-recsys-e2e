## Spec 7: Two-stage LLM rerank (ranking, then explanations) + rerank evaluation

### Problem

`inference/src/steam_inference/rerank.py` asks the LLM, in one `submit_ranking` tool call, to
order all K candidates **and** explain the first `EXPLAIN_TOP_N`. `explanation` is optional on
every entry, so on the first full run (2026-09-25, Nova 2 Lite, 200 reranked users sampled):

- 8% of users have fewer than 5 explanations in their top 5 (the model skipped one or put it on
  entry 6, which is dropped);
- some explanations break the prompt's rule, e.g. "You enjoyed Total War: WARHAMMER II" about the
  candidate being recommended;
- an invalid ranking is silently repaired (missing candidates appended in retrieval order);
- nothing measures whether the LLM order is better than the retrieval order.

### Behaviour

**Stage 1: ranking only.** One call per user returns `{"ranking": [int, ...]}` (candidate
numbers, no text), temperature 0.

- Validated in code: an exact permutation of `1..K`. On failure, one retry that states the error
  (missing / duplicate / unknown numbers); a second failure keeps the retrieval order for that
  user (counted as a rerank failure, as today).
- Position bias: candidates are shown shuffled with a seed derived from the user id
  (reproducible), and mapped back.
- Final order = blend of ranks: `score = w * llm_rank + (1 - w) * retrieval_rank`, ascending,
  ties by retrieval rank. `RERANK_BLEND_WEIGHT` (w, default 1.0 = LLM order; 0 = retrieval order,
  i.e. explanations only).
- `RERANK_ENABLED_STAGES`: `rank+explain` (default) or `explain` (skips stage 1, keeps the
  retrieval order).

**Stage 2: explanations for the final top N only.** A second call with the liked games and only
those N candidates returns exactly N items `{"candidate": int, "text": str}` (`minItems` =
`maxItems` = N, both fields required).

- Checks in code, per explanation:
  - it must not present the candidate as liked or played (the candidate's name after "you
    enjoyed / liked / played" and similar patterns);
  - it must mention at least one liked game name or a genre / developer from the prompt;
  - at most `MAX_EXPLANATION_CHARS`.
- Failed items are re-asked once (only those candidates); a second failure uses a template built
  from the data: "Because you liked {liked game}, which shares {genre or developer}."
- Result: every recommendation in the top N always has an explanation.

**Logging / summary:** per run, counts of stage-1 retries and fallbacks, stage-2 retries and
template explanations, and Bedrock usage per stage.

### Rerank evaluation (decides the defaults)

A command (`python -m steam_inference.evaluate_rerank`) that, for a sample of users active after
the champion's training cutoff (default 500, reranked-eligible: >= `RERANK_MIN_REVIEWS`
reviews):

1. builds each user's history as of the cutoff and retrieves K candidates with the champion;
2. runs stage 1 on them;
3. reports hit rate@5 / @10 against the games they positively reviewed after the cutoff, for:
   retrieval order, LLM order, and blends w in {0.25, 0.5, 0.75};
4. writes the report to `s3://model-artifacts-<acct>/evaluation/rerank/<model_id>/<timestamp>.json`
   and prints a table. Cost bound: the sample size (about $1 per 500 users with Nova 2 Lite).

If no blend beats the retrieval order, the default becomes `explain` only (w = 0).

### Out of scope

Bedrock batch inference (50% off) and changing the default model (Haiku 4.5 can be compared with
the evaluation command by setting `BEDROCK_MODEL_ID`).

### Tests / docs / infra

- Unit tests with a fake `RankFn` / `ExplainFn`: permutation validation and retry, fallback to
  retrieval order, blend and shuffle (reproducible), explanation schema, each grounding check,
  retry of only failed items, template fallback, `explain`-only mode.
- Evaluation test on the synthetic marts (fake LLM that returns a known order).
- The serving contract test still passes (the item shape does not change).
- New settings in `InferenceSettings` + SSM (`infrastructure/inference.tf`): `RERANK_BLEND_WEIGHT`,
  `RERANK_ENABLED_STAGES`, `RERANK_SHUFFLE`.
- `inference/README.md` (how reranking works, the evaluation, the cost per stage),
  `inference/CLAUDE.md` (the invariants: the final order is always a permutation of the
  candidates, every top-N entry has an explanation).

### Implementation notes (2026-09-30)

- The "presents the candidate as liked or played" check is simpler than a verb list: an
  explanation must not name the recommended game at all (it is shown next to it, and every
  "you played <game>" claim names it). Liked-game names are removed first, so a liked
  "Dark Souls III" does not trip the check for a candidate "Dark Souls". The prompt asks for it.
- The grounding check accepts a liked game name, or a genre (also its Spanish name for the
  common Steam genres), tag or developer of the liked games or the candidate.
- If both calls of a user fail (e.g. Bedrock unavailable), the user keeps the retrieval order
  without explanations, as before; if only the explanation call fails, the templates are used.
