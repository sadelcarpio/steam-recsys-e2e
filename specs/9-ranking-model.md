## Spec 9: Ranking model for existing users (batch path)

### Problem

Retrieval is deliberately light: the user tower sees only the last 5 positive reviews
(`user_history_length`), so the online path (5 liked games, `POST /recommendations`) can run it
in the serving Lambda. The champion reaches recall@30 ≈ 12% and recall@100 ≈ 23% on recent
interactions: the right game is often in the top 100 but not in the top 30.

For reviewers with a history (the batch path only), a second stage re-orders the top 100 with
what the two towers cannot use: the full history, negative reviews and user × candidate cross
features. The ceiling is recall@100 of the retriever. The online path is unchanged.

### Data: time split (no leakage)

The retriever's validation window (`timestamp >= cutoff`, `VALIDATION_FRACTION`) is split again
at `T2` (`RANKER_TRAIN_FRACTION` of its rows, default 0.7):

```
            cutoff (T1)                 T2
retriever   train  │  unseen                                │
ranker             │  train (+ early-stopping slice)  │ test │
```

- The ranker learns from candidates of a retriever that never trained on their labels (a
  retriever trained on them would rank the true game near 1, and the ranker would only learn to
  trust the retrieval score).
- One **query group** per positive interaction (u, g, t) in the window: u's history strictly
  before t → retriever → top `RANKER_CANDIDATES` (default 100, already-reviewed games excluded,
  as in inference). Label 1 for g, 0 for the rest. Other games u reviews positively later in the
  window are dropped from the group (not negatives). Training groups without g in the candidates
  are dropped (no positive); test groups keep them (counted as misses).
- Groups sampled per user at most `RANKER_MAX_GROUPS_PER_USER` (default 20), so heavy reviewers
  do not dominate.
- `VALIDATION_FRACTION` must leave enough groups: training fails with a clear message under
  `RANKER_MIN_GROUPS` (default 5000). Recommended `VALIDATION_FRACTION=0.1` when the ranker is on.

### Features (u at time t, candidate c; all as of strictly before t)

From existing marts only (`interactions`, `game_features`, `game_tags`, `game_details`) plus the
retriever's item embeddings; no ETL change.

| Group      | Features                                                                                   |
|------------|--------------------------------------------------------------------------------------------|
| Retrieval  | score, rank                                                                                |
| Candidate  | log positive review count, reviews ratio, is_free, price, release year (parsed)            |
| Affinity   | cosine(u's tag profile, c's weighted tags); share of u's positives with c's developer / publisher; genre / category overlap |
| History    | max and mean embedding similarity of c to u's positives (full history, not only the last 5) |
| Dislikes   | max embedding similarity and tag cosine of c to u's negatively reviewed games              |
| Fit        | c popularity / mean popularity of u's positives; c release year − median of u's positives; c price − median of u's positives |
| User       | positive and negative counts, positive rate, days since last review (only useful through splits with candidate features) |

Profiles are built from u's reviews before t. The same code computes them at training and at
inference (one `ranker.features` module), so there is no train / serve skew.

### Model

LightGBM `lambdarank` (objective NDCG, `label_gain` binary), a few hundred trees, CPU. Params in
`TrainingSettings` (`RANKER_NUM_LEAVES`, `RANKER_LEARNING_RATE`, `RANKER_MAX_ROUNDS`,
early stopping on the last 20% of the training groups by time).

### Training and artifacts

- The training job trains the retriever, then the ranker (`RANKER_ENABLED`, default true), in
  the same job. `python -m steam_training rank --model-id <id>` adds a ranker to an existing
  model (e.g. the current champion) using its cutoff, without retraining it.
- Artifacts, next to the towers: `models/<id>/ranker/model.txt` (LightGBM text dump) and
  `ranker/metadata.json` (`RankerMetadata`: feature names, T1, T2, groups, params, metrics).
- Report on the test window (`t >= T2`), same retriever for both rows:
  recall@{10,30} and NDCG@30 of retrieval order vs ranker order, recall@100 (the ceiling), and
  the top feature importances. Written to `evaluation/<id>/ranker.json` and logged
  (CloudWatch metric definitions).
- **Gate:** the ranker is saved only if its recall@30 beats retrieval order by
  `RANKER_MIN_GAIN` (default 0.005 absolute); otherwise the report says why and the model has no
  ranker.
- Promotion is unchanged (retrieval recall decides, as today). The ranker is copied with its
  model into `models/champion/`; promoting a model without a ranker deletes the champion's copy
  (same rule as `user_tower.npz`), so a ranker is never paired with another retriever.

### Inference

- `RETRIEVAL_CANDIDATES` (default 100) retrieved; when `models/champion/ranker/` exists and
  `RANKER_ENABLED`, the ranker re-orders them and the top `TOP_K` (30) is kept, else the
  retrieval top `TOP_K` as today.
- The LLM rerank / explanations run on the ranker's top K, unchanged (spec 7 can then default to
  explain-only).
- Only users with at least `RANKER_MIN_REVIEWS` (default 1) reviews are ranked; the popularity
  fallback and the DynamoDB item shape do not change. The summary logs ranker time and users
  ranked.

### Out of scope

Online ranking (serving stays retrieval + client-side filters), review text features, playtime
(`playtime_at_review`, `num_games_owned` are scraped but dropped by `stg_steam__reviews`: a
follow-up adds them to the marts, then as features / graded labels), refitting the retriever on
the full data after the ranker is trained.

### Tests / docs / infra

- Unit tests: group building (label, dropped groups, per-user cap, no feature from t or later),
  each feature family on hand-built histories, the gate, artifact save / load / promotion copy
  and deletion.
- End to end on the synthetic marts (`conftest.py`): the ranker beats retrieval order on data
  with a signal the towers cannot see (e.g. users who avoid a disliked developer).
- Inference: with and without a champion ranker; the serving contract test still passes.
- `uv add lightgbm` in `training` and `inference` (images only; the serving Lambda is untouched).
- New settings in `TrainingSettings` / `InferenceSettings` + SSM (`infrastructure/training.tf`,
  `infrastructure/inference.tf`).
- `training/README.md`, `training/CLAUDE.md` (invariants: ranker features only from before t,
  ranker always paired with its retriever), `inference/README.md`, `inference/CLAUDE.md`,
  `docs/deployment.md` (adding a ranker to the current champion).
