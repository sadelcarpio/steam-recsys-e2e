## Spec 9: Ranking model for existing users (batch path)

### Problem

The retrieval champion (`local-9cc8927-a3-len20-attn`, spec 17 A3 on top of A4) encodes the
user as the last 20 positive reviews (`HISTORY_LENGTH=20`, `HISTORY_ENCODER=attention`: masked
mean + one self-attention block with learned positions and a summary token) plus a tag profile
of that history (`USER_TAG_PROFILE`). The same tower runs online (up to 20 liked games,
`POST /recommendations`, numpy format 3 in the serving Lambda). On its validation split (after
the 2026-05-01 cutoff) it reaches, on warm rows, recall@30 ≈ 17%, recall@50 ≈ 22% and
recall@100 ≈ 30%: the right game is in the top 100 almost twice as often as in the top 30,
and the final list is shorter still (20).

For reviewers with a history (the batch path only), the pipeline becomes two stages: retrieval
returns ~100 candidates, and a ranker re-orders them and keeps the top 20, using what the two
towers cannot use:

- the history beyond the last 20 positives (heavy reviewers have hundreds);
- negative reviews as a user signal (the retriever only uses them as training hard negatives;
  the user tower never sees them);
- explicit user × candidate cross features (developer / publisher loyalty, price, release year
  and popularity fit), which a dot product of two independent towers can only approximate.

The ceiling is recall@100 of the retriever. The online path is unchanged.

### Data: time split (no leakage)

The retriever's validation window (`timestamp >= cutoff`, `VALIDATION_FRACTION`, 0.1 for the
champion: ~12M rows after 2026-05-01) is split again at `T2` (`RANKER_TRAIN_FRACTION` of its
rows, default 0.7):

```
            cutoff (T1)                 T2
retriever   train  │  unseen                                │
ranker             │  train (+ early-stopping slice)  │ test │
```

- The ranker learns from candidates of a retriever that never trained on their labels: one
  trained on them would almost always return the true game near the top, and the ranker would
  learn an easier problem than the one it faces at serving time (the future, where the true game
  is often missing or low). This holds whatever the features are: the candidate list itself is
  the retriever's output.
- One **query group** per positive interaction (u, g, t) in the window with at least one
  positive review of u before t (warm rows: inference never scores an empty history; cold rows,
  ~30% of the window, stay a retrieval / popularity problem). u's history strictly before t →
  retriever (the model's own `history_length` and encoder, histories derived from
  `interactions` with the training `HistoryIndex`, as in training and inference) → top
  `RANKER_CANDIDATES` (default 100, already-reviewed games excluded, as in inference). Label 1
  for g, 0 for the rest. Other games u reviews positively later in the window are dropped from
  the group (not negatives). Training groups without g in the candidates are dropped (no
  positive); test groups keep them (counted as misses).
- Groups sampled per user at most `RANKER_MAX_GROUPS_PER_USER` (default 20), so heavy reviewers
  do not dominate.
- `VALIDATION_FRACTION` must leave enough groups: training fails with a clear message under
  `RANKER_MIN_GROUPS` (default 5000). The default 0.1 is enough on the current data.

### Features (u at time t, candidate c; all as of strictly before t)

From existing marts only (`interactions`, `game_features`, `game_tags`, `game_details`); no ETL
change. **No feature depends on a retriever version** (no raw score, no tower embeddings): the
retrieval stage enters only through scale-free positions, so a retriever can be retrained and
promoted without retraining the ranker (see "MLOps").

| Group      | Features                                                                                   |
|------------|--------------------------------------------------------------------------------------------|
| Retrieval  | rank, score percentile in the group, score − the group's top score divided by the group's score spread (scale-free) |
| Candidate  | log positive reviews as of t (lifetime, last 30 days, last 90 days), reviews ratio, is_free, price, game age at t (days since release) |
| Affinity   | cosine(u's tag profile, c's weighted tags), over the full history and over the tower's last 20; share of u's positives with c's developer / publisher; genre / category overlap |
| History    | max and mean tag cosine of c to each of u's positives: full history and the older part beyond the tower's last 20 (separately, so the ranker can tell what the tower already saw) |
| Negatives  | tag cosine of c to u's negatively reviewed games (an interest signal, not avoidance: see below) |
| Fit        | c popularity / mean popularity of u's positives; c release date − median of u's positives; c price − median of u's positives |
| User       | positive and negative counts, positive rate, days since last review, history longer than the tower's `history_length` (only useful through splits with candidate features) |

Developer / publisher / genre enter as user × candidate interactions (shares, overlaps), not as
embeddings: a tree splits one dimension at a time and an embedding's axes mean nothing (and move
on every retrain); the retriever already consumes those embeddings, and what the ranker adds is
"this user keeps reviewing this developer". Learned embeddings in the ranker are the neural
follow-up's business (see "Model"), through a pinned embedding artifact, never borrowed from the
current retriever.

Evidence (`notebooks/02_ranking_features`, public dataset, a proxy retriever since the towers
don't run there; family ranking, not absolute gains):

- Recent popularity as of t is the strongest single feature (30-day count: within-group AUC
  0.67, top gain); lifetime popularity 0.57. The towers can't learn it (logQ correction removes
  popularity on purpose), so it belongs to the ranker. Computed from `interactions`, no ETL change.
- Similarity to negatively reviewed games *raises* the odds (tag cosine AUC 0.61), as in spec 17
  Q7: negative reviews sit in the user's taste region. The ranker learns the sign; the family is
  not a "dislike" filter. Embedding similarity and developer share of negatives add nothing.
- Tag profile cosine is the main affinity feature (AUC 0.65; weaker with the champion, whose
  user tower already reads a tag profile). History beyond the last 20 has some signal (AUC 0.56)
  but only for the ~8% of groups that have it. Users with more than 20 positives are where
  retrieval is worst and the ranker gains most.
- Most of the ranker's value is in retriever-independent families (drop one, Δ recall@20:
  candidate −0.0066, language −0.0053, negatives −0.0029; embedding history −0.0013; the raw score
  ~2% of gain): decoupling costs an estimated 0.001–0.003 recall@20, to re-check on the champion.
- Game age at t beats release year; developer loyalty is weak per candidate (few candidates share
  a developer with the history).

Profiles are built from u's reviews before t. The same code computes them at training and at
inference (one `ranker.features` module), so there is no train / serve skew. The tower's history
length is read from the retriever's `ModelConfig`, never hard-coded. `FEATURE_SCHEMA_VERSION`
(in `RankerMetadata`) changes whenever a feature's definition does.

### Model

LightGBM `lambdarank` (objective NDCG, `label_gain` binary), a few hundred trees, CPU. Params in
`TrainingSettings` (`RANKER_NUM_LEAVES`, `RANKER_LEARNING_RATE`, `RANKER_MAX_ROUNDS`,
early stopping on the last 20% of the training groups by time).

Why trees first, not a neural ranker (DLRM, Wide & Deep, DCN, DIN / BST):

- The inputs are ~35 dense, heterogeneous hand-built features; trees need no scaling and remain
  hard to beat on mid-size tabular data (Grinsztajn et al. 2022; Airbnb, "Applying Deep Learning
  to Airbnb Search", 2018, where a GBDT baseline took sustained work to overtake).
- Data is modest: ~300k query groups on the full data. Neural rankers pay off at far more.
- The neural parts (game / developer / genre ids, attention over the sequence) already live in
  the retriever (A3 is a SASRec-style encoder). DLRM / Wide & Deep / DCN over the same ids would
  mostly re-learn its embeddings, and overfit at this size.
- CPU training in minutes (the single `g4dn.xlarge` GPU stays with the retriever), batch scoring
  of ~20M rows (200k users × 100) in minutes, a small `model.txt`, readable importances.
- It is the baseline any neural ranker must beat under the same gate.

**Neural follow-up (own spec):** a target-attention ranker (DIN / BST) that attends over the
user's *full* history with the candidate as the query, plus the same dense features as a wide
part. It targets the measured weak spot (long histories, where trees only see max / mean
summaries). SASRec / BERT4Rec are next-item (retrieval) models, already covered by the retriever.
The follow-up would read learned item / attribute embeddings from a pinned artifact
(`embeddings/<version>/`, its own lifecycle) so it stays decoupled from retriever retrains.

### Training and artifacts

- The retriever and the ranker train in **separate jobs** with separate lifecycles. `python -m
  steam_training rank --retriever-id <id>` builds the groups from that retriever's candidates
  (its cutoff is T1) and trains a ranker; default `<id>` = the champion
  (`local-9cc8927-a3-len20-attn`, cutoff 2026-05-01). CPU only (a processing / training job on
  `ml.m5.2xlarge`), no GPU.
- Artifacts, independent of any retriever: `rankers/<ranker_id>/model.txt` (LightGBM text dump)
  and `rankers/<ranker_id>/metadata.json` (`RankerMetadata`: feature names,
  `FEATURE_SCHEMA_VERSION`, the retriever its candidates came from, T1, T2, groups, params,
  metrics, Iceberg snapshot ids). `ranker_id` follows the model id convention (`<sha>` from CD,
  `local-<sha>-<suffix>` locally).
- Report on the test window (`t >= T2`), same retriever for both rows: recall@{10,20} and
  NDCG@20 of retrieval order vs ranker order, recall@100 (the ceiling), the top feature
  importances, the drop-one-family deltas, and a split by history length (≤ 20 positives vs
  more). Written to `evaluation/rankers/<ranker_id>/ranker.json` and logged (CloudWatch metric
  definitions, MLflow when configured, spec 18).
- **Ranker gate:** a ranker is a challenger only if its recall@20 beats retrieval order by
  `RANKER_MIN_GAIN` (default 0.005 absolute); otherwise the report says why.

### Inference

- `RETRIEVAL_CANDIDATES` (default 100) retrieved with the champion's tower (history length and
  encoder from its metadata); when a champion ranker exists, `RANKER_MODE` is `on` and the
  serving pair is valid (see "MLOps"), the ranker re-orders them and the top `RANKER_TOP_K`
  (default 20) is kept, else the retrieval top `TOP_K` (30) as today. `EXPLAIN_TOP_N` must be ≤
  `RANKER_TOP_K`.
- The ranker's profiles reuse the reviews inference already derives the tower histories from
  (full history, positives and negatives), for the users scored in the run (the 200k most
  active, spec 14).
- The LLM rerank / explanations run on the ranker's top 20, unchanged (spec 7 can then default to
  explain-only).
- Only users with at least `RANKER_MIN_REVIEWS` (default 1) reviews are ranked; the popularity
  fallback and the DynamoDB item shape do not change. A ranker failure on a batch falls back to
  retrieval order for that batch (logged, the run goes on). The summary logs ranker time, users
  ranked and overlap@20 between retrieval and ranker order.

### MLOps: two models, one evaluated pair

The ranker still depends on the retriever through the candidate list it re-orders, but none of
its features depend on a retriever version, so the two have **independent lifecycles**. The one
rule: **never serve a (retriever, ranker) pair that wasn't evaluated together.** It is enforced by
evaluation at promotion, not by retraining.

- **Versioning.** `models/<id>/` (retrievers) and `rankers/<ranker_id>/` are immutable once
  written (bucket versioning is on). `models/champion/` and `rankers/champion/` hold the current
  copies; `evaluation/champion/pair.json` names the serving pair (retriever id, ranker id or none,
  the window and metrics it was evaluated on). Inference refuses a champion ranker that is not
  the one in `pair.json` (writes retrieval order and logs why). MLflow (spec 18): registered
  models `steam-two-tower` and `steam-ranker`, each with `champion` / `challenger` aliases; the
  ranker run is tagged with the retriever its candidates came from.
- **Lifecycles.**
  - New retriever → retriever gate (as today: warm recall@50 vs champion), plus warm
    recall@100 >= champion (the ranker's ceiling never drops), then a **compatibility check**: the
    champion ranker re-orders the new retriever's candidates on a window both have not trained
    on (no retraining). If the ranker still beats retrieval order by `RANKER_MIN_GAIN`, the new
    pair is promoted with it; if not, the retriever is promoted alone (`pair.json` with no ranker)
    and a ranker retrain is due.
  - New ranker (features, params, a newer window) → `rank` against the champion retriever →
    ranker gate → final-list gate vs the champion pair.
  - Both at once (a full retrain) is the same two steps in order.
- **Final-list gate** (every promotion that changes the serving pair): warm recall@20 and
  NDCG@20 of the new pair's top 20 beat the champion pair's by `MIN_IMPROVEMENT`, on one common
  window both pairs never trained on (t >= the newer ranker's T2 and after both retrievers'
  cutoffs). With one held-out game per group, precision@20 = recall@20 / 20, so the recall gate is
  also the precision gate. The report adds top-20 churn vs the champion (share of users' lists
  that change).
- **Promote / rollback.** Promotion copies the changed model(s) into their `champion/` folders,
  writes `pair.json`, and appends `evaluation/champion/history.jsonl` (one line per promotion:
  pair, metrics, actor, time). Rollback = `training promote ROLLBACK=1` (restores the previous
  line's pair; either model alone or both), then an inference run (DynamoDB keeps the current
  lists until the next run). `RANKER_MODE=off` (SSM) is the no-deploy kill switch: the next
  inference run writes retrieval order.
- **Shadow first.** `RANKER_MODE=shadow` (default for the first runs): inference computes the
  ranker order, logs overlap@20 with retrieval order and ranker time, but writes retrieval order.
  `RANKER_MODE=on` after one clean run.
- **Delayed-label check.** Each inference run scores the previous run's lists against the reviews
  that arrived since (recall@20 of what was written; retrieval order vs ranker order in shadow),
  as `evaluation/online/<run_id>.json` + CloudWatch metrics: the batch path's online metric.
- **Data window.** A ranker trains on a window its candidate retriever never saw, and every
  compatibility / final-list evaluation needs a window none of the evaluated models saw. A
  retriever refit on all data (`FIT_ON=all`) leaves no such window, so it is refused while a
  ranker serves. Rolling windows (each cycle moves T1 forward: retriever ≤ T1, ranker T1..now)
  are the follow-up.
- **Orchestration.** Two pipelines, not one job: retriever training (GPU) → retriever gate →
  compatibility check; ranker training (CPU) → ranker gate → final-list gate. A training Step
  Function chaining them (with a manual approval before the champion changes, as for deploys) is
  a follow-up; until then both are manual workflows (`training-cd.yml`, a new
  `training-rank.yml`, `training-promote.yml`).

### Out of scope

Online ranking (serving stays retrieval + client-side filters), review text features,
refitting the retriever on the full data while a ranker serves, cold (empty-history) users, the
neural ranker (see "Model"), the training Step Function.

Follow-ups that need an ETL change (the columns are scraped but dropped by `stg_steam__reviews`):

- **Reviewer language** (first): the share of c's reviews in u's dominant language. Second most
  useful family in the EDA (AUC 0.61; dropping it costs ~0.5 points of recall@20, more than any
  family but Candidate). Needs `language` in `interactions`.
- **Playtime** (`playtime_at_review`): a playtime-weighted tag profile scores the same as the
  count-weighted one (AUC 0.649 vs 0.648), so not worth it as a feature; only graded labels (hours
  played on the target) remain untested.

### Tests / docs / infra

- Unit tests: group building (label, dropped groups, warm-only groups, per-user cap, no feature
  from t or later), each feature family on hand-built histories (including the beyond-20 split
  with a small `history_length`), features unchanged when the retriever's raw scores are rescaled
  (scale-free retrieval features), the ranker gate, artifact save / load.
- Unit tests, promotion: retriever-only promotion keeps the champion ranker when the
  compatibility check passes and drops it when it fails, ranker-only promotion, the final-list
  gate, `pair.json` / `history.jsonl`, `ROLLBACK=1` for each case.
- Inference: with and without a champion ranker, a ranker not in `pair.json` is ignored, shadow
  mode writes retrieval order, a ranker failure falls back per batch; with an attention champion;
  the serving contract test still passes.
- End to end on the synthetic marts (`conftest.py`): the ranker beats retrieval order on data
  with a signal the towers cannot see (e.g. a burst of recent popularity).
- `uv add lightgbm` in `training` and `inference` (images only; the serving Lambda is untouched).
- New settings in `TrainingSettings` / `InferenceSettings` + SSM (`infrastructure/training.tf`,
  `infrastructure/inference.tf`); `training-rank.yml` (manual, like `training-cd.yml`).
- Re-check the feature families on the champion's real candidates (the EDA used a proxy
  retriever), including the cost of the decoupled retrieval features vs the raw score.
- `training/README.md`, `training/CLAUDE.md` (invariants: ranker features only from before t, no
  ranker feature depends on a retriever version, never serve a pair not evaluated together),
  `inference/README.md`, `inference/CLAUDE.md`, `docs/deployment.md` (adding a ranker to the
  current champion, rollback).
