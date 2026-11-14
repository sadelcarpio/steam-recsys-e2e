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

- The ranker learns from candidates of a retriever that never trained on their labels (a
  retriever trained on them would rank the true game near 1, and the ranker would only learn to
  trust the retrieval score).
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

From existing marts only (`interactions`, `game_features`, `game_tags`, `game_details`) plus the
retriever's item embeddings (item tower outputs); no ETL change.

| Group      | Features                                                                                   |
|------------|--------------------------------------------------------------------------------------------|
| Retrieval  | score, rank                                                                                |
| Candidate  | log positive reviews as of t (lifetime, last 30 days, last 90 days), reviews ratio, is_free, price, game age at t (days since release) |
| Affinity   | cosine(u's tag profile, c's weighted tags), over the full history and over the tower's last 20; share of u's positives with c's developer / publisher; genre / category overlap |
| History    | max and mean embedding similarity of c to u's positives: full history and the older part beyond the tower's last 20 (separately, so the ranker can tell what the tower already saw) |
| Negatives  | tag cosine of c to u's negatively reviewed games (an interest signal, not avoidance: see below) |
| Fit        | c popularity / mean popularity of u's positives; c release date − median of u's positives; c price − median of u's positives |
| User       | positive and negative counts, positive rate, days since last review, history longer than the tower's `history_length` (only useful through splits with candidate features) |

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
- Game age at t beats release year; developer loyalty is weak per candidate (few candidates share
  a developer with the history).

Profiles are built from u's reviews before t. The same code computes them at training and at
inference (one `ranker.features` module), so there is no train / serve skew. The tower's history
length is read from the retriever's `ModelConfig`, never hard-coded.

### Model

LightGBM `lambdarank` (objective NDCG, `label_gain` binary), a few hundred trees, CPU. Params in
`TrainingSettings` (`RANKER_NUM_LEAVES`, `RANKER_LEARNING_RATE`, `RANKER_MAX_ROUNDS`,
early stopping on the last 20% of the training groups by time).

### Training and artifacts

- The training job trains the retriever, then the ranker (`RANKER_ENABLED`, default true), in
  the same job. The retriever's training rows are released before the ranker builds its groups
  (the full load already peaks at ~14 GB on the 16 GB `g4dn.xlarge`). `python -m steam_training
  rank --model-id <id>` adds a ranker to an existing model (e.g. the current champion
  `local-9cc8927-a3-len20-attn`, cutoff 2026-05-01) using its cutoff and config, without
  retraining it.
- Artifacts, next to the towers: `models/<id>/ranker/model.txt` (LightGBM text dump) and
  `ranker/metadata.json` (`RankerMetadata`: feature names, retriever `model_id` and
  `history_length`, T1, T2, groups, params, metrics).
- Report on the test window (`t >= T2`), same retriever for both rows:
  recall@{10,20} and NDCG@20 of retrieval order vs ranker order, recall@100 (the ceiling), and
  the top feature importances; also split by history length (≤ 20 positives vs more), since the
  ranker's extra history only exists for the second group. Written to
  `evaluation/<id>/ranker.json` and logged (CloudWatch metric definitions, MLflow when
  configured, spec 18).
- **Gate:** the ranker is saved only if its recall@20 beats retrieval order by
  `RANKER_MIN_GAIN` (default 0.005 absolute); otherwise the report says why and the model has no
  ranker.
- Promotion is unchanged (retrieval warm recall@50 decides, as today). The ranker is copied with
  its model into `models/champion/`; promoting a model without a ranker deletes the champion's
  copy (same rule as `user_tower.npz`), so a ranker is never paired with another retriever.
  Inference also checks that `ranker/metadata.json` names the champion's `model_id`.

### Inference

- `RETRIEVAL_CANDIDATES` (default 100) retrieved with the champion's tower (history length and
  encoder from its metadata); when `models/champion/ranker/` exists and `RANKER_ENABLED`, the
  ranker re-orders them and the top `RANKER_TOP_K` (default 20) is kept, else the retrieval top
  `TOP_K` (30) as today. `EXPLAIN_TOP_N` must be ≤ `RANKER_TOP_K`.
- The ranker's profiles reuse the reviews inference already derives the tower histories from
  (full history, positives and negatives), for the users scored in the run (the 200k most
  active, spec 14).
- The LLM rerank / explanations run on the ranker's top 20, unchanged (spec 7 can then default to
  explain-only).
- Only users with at least `RANKER_MIN_REVIEWS` (default 1) reviews are ranked; the popularity
  fallback and the DynamoDB item shape do not change. The summary logs ranker time and users
  ranked.

### Out of scope

Online ranking (serving stays retrieval + client-side filters), review text features,
refitting the retriever on the full data after the ranker is trained, cold (empty-history) users.

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
  with a small `history_length`), the gate, artifact save / load / promotion copy and deletion,
  a ranker whose `model_id` does not match the champion is ignored.
- End to end on the synthetic marts (`conftest.py`): the ranker beats retrieval order on data
  with a signal the towers cannot see (e.g. users who avoid a disliked developer).
- Inference: with and without a champion ranker, with an attention champion; the serving
  contract test still passes.
- `uv add lightgbm` in `training` and `inference` (images only; the serving Lambda is untouched).
- New settings in `TrainingSettings` / `InferenceSettings` + SSM (`infrastructure/training.tf`,
  `infrastructure/inference.tf`).
- Re-check the feature families on the champion's real candidates (the EDA used a proxy
  retriever) and report the drop-one-family deltas in `ranker.json`.
- `training/README.md`, `training/CLAUDE.md` (invariants: ranker features only from before t,
  ranker always paired with its retriever), `inference/README.md`, `inference/CLAUDE.md`,
  `docs/deployment.md` (adding a ranker to the current champion).
