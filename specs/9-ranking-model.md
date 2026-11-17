## Spec 9: Ranking model (batch and online paths)

### Problem

The retrieval champion (`local-9cc8927-a3-len20-attn`, spec 17 A3 on top of A4) encodes the
user as the last 20 positive reviews (`HISTORY_LENGTH=20`, `HISTORY_ENCODER=attention`: masked
mean + one self-attention block with learned positions and a summary token) plus a tag profile
of that history (`USER_TAG_PROFILE`). The same tower runs online (up to 20 liked games,
`POST /recommendations`, numpy format 3 in the serving Lambda). On its validation split (after
the 2026-05-01 cutoff) it reaches, on warm rows, recall@30 ≈ 17%, recall@50 ≈ 22% and
recall@100 ≈ 30%: the right game is in the top 100 almost twice as often as in the top 30,
and the final list is shorter still (20).

The pipeline becomes two stages: retrieval returns ~100 candidates, and a ranker re-orders
them and keeps the top 20. **One ranker serves both paths**: the batch path (reviewers with a
history, precomputed into DynamoDB) and the online path (`POST /recommendations`, an anonymous
visitor's liked games, the app's main use case). It adds what the two towers cannot express:

- explicit user × candidate cross features (tag affinity, developer / publisher loyalty, price,
  release date and popularity fit), which a dot product of two independent towers can only
  approximate;
- popularity as of now (the last 30 / 90 days): the towers can't learn it (the logQ correction
  removes popularity on purpose), and Steam popularity moves fast.

To run online, every user-side feature is computed from what a request carries: at most 20 liked
games (`MAX_LIKED_GAMES`), most recent first. Batch users are treated the same way (their last 20
positive reviews), so one model, one gate and one lifecycle cover both paths. The ceiling is
recall@100 of the retriever. Visitors with no liked game are a cold-start problem (see "Cold
start"), not a ranking one.

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

H = u's last 20 positive reviews before t (most recent first; online: the request's liked games,
in the order given). Every user-side feature is a function of H only, so the same code runs in
training, batch inference and the serving Lambda. **No feature depends on a retriever version**
(no raw score, no tower embeddings): retrieval enters only through scale-free positions, so a
retriever can be retrained and promoted without retraining the ranker (see "MLOps").

| Group      | Features                                                                                   |
|------------|--------------------------------------------------------------------------------------------|
| Retrieval  | rank, score percentile in the group, (score − the group's top score) / the group's score spread |
| Candidate  | log positive reviews as of t (lifetime, last 30 days, last 90 days), smoothed positive ratio as of t, is_free, log price, game age at t (days since release) |
| Affinity   | cosine(tag profile of H, c's weighted tags); share of H with c's developer / publisher; genre / category overlap of c with H |
| Similarity | max and mean tag cosine of c to each game of H                                              |
| Fit        | c log popularity − mean of H's; c release date − median of H's (years); c log price − median of H's; share of free games in H |
| User       | \|H\| (only useful through splits with candidate features)                                 |
| Language   | share of c's reviews in u's language (batch: u's dominant review language; online: the browser locale). Needs `language` in `interactions` (ETL follow-up below) |

Batch-only signals the EDA measured are left out to keep one model for both paths: similarity to
negatively reviewed games (−0.003 recall@20 when dropped), history older than the last 20
(−0.001) and user activity features (−0.001). A batch-only extension can add them back as a
second model if a gate shows it is worth one.

Developer / publisher / genre enter as user × candidate interactions (shares, overlaps), not as
embeddings: a tree splits one dimension at a time and an embedding's axes mean nothing (and move
on every retrain); the retriever already consumes those embeddings, and what the ranker adds is
"this user keeps liking this developer". Learned embeddings in the ranker are the neural
follow-up's business (see "Model"), through a pinned embedding artifact, never borrowed from the
current retriever.

**Popularity is a mart, defined once.** `steam_marts.game_popularity_weekly` (game_id, game_idx,
week, positives, reviews), incremental, point-in-time by construction. Consumers: ranker training
(as-of lookups: the weeks strictly before t's week), batch inference (the latest weeks: ranker
features, the `__popular__` list, spec 15's popularity), and the online export (the current
game-side table). Until it ships, training computes the same aggregation from `interactions`
(same definition, `ranker.popularity`), so switching is a source change.

Evidence (`notebooks/02_ranking_features`, public dataset, a proxy retriever since the towers
don't run there; family ranking, not absolute gains):

- Recent popularity as of t is the strongest single feature (30-day count: within-group AUC
  0.67, top gain); lifetime popularity 0.57.
- Tag profile cosine is the main affinity feature (AUC 0.65; weaker with the champion, whose
  user tower already reads a tag profile).
- Language is the second most useful family (AUC 0.61, −0.005 recall@20 when dropped).
- Most of the ranker's value is in retriever-independent families; decoupling from the raw score
  costs an estimated 0.001–0.003 recall@20, to re-check on the champion.
- On the proxy, groups with ≤ 20 positives (the online case) went from 0.098 to 0.137 recall@20.
- Game age at t beats release year; developer loyalty is weak per candidate.

The same code computes the features at training and at inference (`steam_training.ranker.
features`, pure numpy, imported by inference; the serving Lambda has a numpy port with a parity
test). `FEATURE_SCHEMA_VERSION` (in `RankerMetadata`) changes whenever a feature's definition
does. `game_price` is the price at scrape time (not as of t) in training.

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

Batch (`steam_inference`):

- `RETRIEVAL_CANDIDATES` (default 100) retrieved with the champion's tower (history length and
  encoder from its metadata); when a champion ranker exists, `RANKER_MODE` is `on` and the
  serving pair is valid (see "MLOps"), the ranker re-orders them and the top `RANKER_TOP_K`
  (default 20) is kept, else the retrieval top `TOP_K` (30) as today. `EXPLAIN_TOP_N` must be ≤
  `RANKER_TOP_K`.
- H = the user's last 20 positive reviews (inference already derives them for the tower), for
  the users scored in the run (the 200k most active, spec 14); popularity = the latest complete
  weeks of the mart.
- The LLM rerank / explanations run on the ranker's top 20, unchanged (spec 7 can then default to
  explain-only).
- A ranker failure on a batch falls back to retrieval order for that batch (logged, the run goes
  on). The summary logs ranker time, users ranked and overlap@20 between retrieval and ranker
  order. The popularity fallback and the DynamoDB item shape do not change.
- Every run also exports the game-side feature table next to the online catalog
  (`serving/online/game_features.npz`: per catalog game the candidate features as of the run,
  tags, developer / publisher / genre / category ids), named in the online manifest.

Online (`serving`, `POST /recommendations`):

- Retrieval top 100 (as today, exact top K over the online catalog), then the ranker: user
  features from the request's liked games, game features from `game_features.npz` (loaded at
  cold start like the catalog), language from `Accept-Language`, then the top 20.
- The ranker runs as a **numpy tree evaluator** (`rankers/<id>/ranker.npz`: per tree the split
  feature, threshold, children, leaf values, exported from the LightGBM dump), so the Lambda
  needs no LightGBM. Parity with LightGBM's predictions is tested (as `test_online_parity.py`
  does for the user tower). Budget: 100 candidates × a few hundred trees, a few ms.
- Response `source` stays `online`; the ranker id is logged. No ranker (or `RANKER_MODE` not
  `on`, or a pair mismatch): retrieval order, as today.

### Cold start (no liked game)

Not a ranking problem: there is no H. Evidence (2026-11-12, warm vs cold rows after the
champion's cutoff): on empty histories, popularity of the last 2–26 weeks reaches recall@50
0.30–0.32 vs 0.17 for the champion's constant cold query and 0.175 for all-time popularity.

- `/popular` and the `__popular__` fallback stay recent popularity (`POPULAR_WINDOW_DAYS`),
  read from the popularity mart once it ships.
- Next (with the language ETL): one popular list per language (`__popular__#<language>`),
  chosen from `Accept-Language`, falling back to the global list. Language is the cheapest
  personal signal an anonymous visitor carries.
- The app asks for liked games early (the frontend already does): from one liked game on, the
  visitor is on the ranked path.

### MLOps: two models, one evaluated pair

The ranker still depends on the retriever through the candidate list it re-orders, but none of
its features depend on a retriever version, so the two have **independent lifecycles**. The one
rule: **never serve a (retriever, ranker) pair that wasn't evaluated together.** It is enforced by
evaluation at promotion, not by retraining.

- **Versioning.** `models/<id>/` (retrievers) and `rankers/<ranker_id>/` are immutable once
  written (bucket versioning is on). `models/champion/` and `rankers/champion/` hold the current
  copies; `evaluation/champion/pair.json` names the serving pair (retriever id, ranker id or none,
  the window and metrics it was evaluated on). Inference refuses a champion ranker that is not
  the one in `pair.json` (writes retrieval order and logs why); the online manifest names the
  pair too, and serving applies the same rule. MLflow (spec 18): registered
  models `steam-two-tower` and `steam-ranker`, each with `champion` / `challenger` aliases; the
  ranker run is tagged with the retriever its candidates came from.
- **Lifecycles.**
  - New retriever → retriever gate (as today: warm recall@50 vs champion), plus warm
    recall@100 >= champion (the ranker's ceiling never drops). Then two candidate pairs are
    evaluated on a window none of the models saw: (new retriever, champion ranker) and (new
    retriever, a ranker retrained on the new retriever's candidates; minutes on CPU). The better
    pair is promoted if it passes the final-list gate against the champion pair; otherwise
    nothing is promoted. A retriever that wins at retrieval but loses as a pair is not served.
    The first pair's result says whether the champion ranker generalises to the new candidates
    (reported, not a branch).
  - New ranker (features, params, a newer window) → `rank` against the champion retriever →
    ranker gate → final-list gate vs the champion pair.
  - Both at once (a full retrain) is the same two steps in order.
- **Final-list gate** (every promotion that changes the serving pair): warm recall@20 and
  NDCG@20 of the new pair's top 20 beat the champion pair's by `MIN_IMPROVEMENT`, on one common
  window both pairs never trained on (t >= the newer ranker's T2 and after both retrievers'
  cutoffs). With one held-out game per group, precision@20 = recall@20 / 20, so the recall gate is
  also the precision gate. The report adds top-20 churn vs the champion (share of users' lists
  that change).
- **Promote / rollback.** Promotion copies the changed model(s) into their `champion/` folders
  (the ranker with its `ranker.npz`),
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
- **Data window** (details and diagrams: "Time windows"). A ranker trains on a window its
  candidate retriever never saw, and every compatibility / final-list evaluation needs a window
  none of the evaluated models saw. A retriever trained through N (the serving retriever R_B) is
  therefore never used for ranker data or gates: it only exists as the refreshed twin of an
  evaluated retriever R_A (pass B), and only after R_A passed.
- **Orchestration.** Two pipelines, not one job: retriever training (GPU) → retriever gate →
  pair evaluation; ranker training (CPU) → ranker gate → final-list gate. Manual workflows first
  (`training-cd.yml`, a new `training-rank.yml`, `training-promote.yml`), then a training Step
  Function (see "Delivery sequence").
- **Without a live MLflow server.** S3 is the registry: immutable `models/<id>/` and
  `rankers/<id>/`, the `champion/` copies, `pair.json`, `history.jsonl` and every report under
  `evaluation/`. MLflow stays optional: runs are replayed into it (`track`), and promotion sets
  the `champion` / `challenger` aliases when `MLFLOW_TRACKING_URI` is set.

### Time windows: challengers and continuous training

The retriever's cutoff is a share of rows today (`VALIDATION_FRACTION=0.1`, ~6.5 months after
2026-05-01 for the champion), far more than a ranker needs (~450k reviews a week: 4 weeks give
~1M warm groups). Phase 3 replaces it with **time-based windows counted back from the latest
data N** (`HOLDOUT_WEEKS`, default 6; `TEST_WEEKS`, default 2). Every window below is
`[start, end)`.

**Rules** (every case below follows from them):

1. A retriever never trains on anything at or after its cutoff.
2. A ranker trains only on candidates of a retriever that never saw the ranker's window (else the
   true game sits near the top and the ranker learns an easier problem than serving).
3. Every comparison with the champion pair uses a window **none** of the compared models (either
   retriever, either ranker) trained on: the test window `[N − 2w, N)`.
4. Validation (epoch selection, early stopping) may overlap another model's training window, but
   never the test window: the test window is only read by the gates.

#### 1. One pair from scratch (the happy path)

```
time ──────────────────────────────────────────────────────────────────────────►
                                           N − 6w              N − 2w        N
retriever train   ██████████████████████████│
retriever valid                             │░░░░░░░░░░░░░░░░░░│              (epochs, monitor)
ranker train                                │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│              (last 20%: early stop)
test (gates)                                                   │▒▒▒▒▒▒▒▒▒▒▒▒▒│
```

- **Retriever**: trains on everything before `N − 6w`; picks its epoch on `[N − 6w, N − 2w)`.
- **Ranker**: the retriever's candidates on `[N − 6w, N − 2w)` (the same window: the retriever
  never trained on those labels, it only selected an epoch with them, rule 4); its last 20% by
  time stops boosting.
- **Test** `[N − 2w, N)`: retrieval recall@50 / @100, ranker vs retrieval order, the pair's
  recall@20 / NDCG@20. Nothing trained on it.
- **Serving**: see the refresh step in 2 (the pair above is the *evaluated* pair; the served one
  also learns the last 6 weeks).

#### 2. Continuous training (monthly, the champion's lineage)

Two lineages of the same retriever, one step apart:

- the **evaluation retriever** R_A: always stops at `N − 6w`, so the latest 6 weeks stay unseen
  (rankers train on its candidates; gates use it);
- the **serving retriever** R_B: R_A fine-tuned on those 6 weeks, trained through N. It serves,
  but it can never produce ranker training data (it saw every window) and it is never the
  starting point of the next cycle (next month's ranker window overlaps what it saw).

Cycle n (run date N; previous run N' ≈ N − 4w; R_A' / R_B' / K' = the previous cycle's models):

```
time ───────────────────────────────────────────────────────────────────────────────────────────►
                                     N'−6w             N−6w                    N−2w             N
R_A' (previous evaluation)        ██████│                 │                       │             │
pass A  R_A = R_A' + delta        ██████│░░░░░░░░░░░░░░░░░│                       │             │  warm start + replay, ~1 epoch
        R_A validation                  │                 │░░░░░░░░░░░░░░░░░░░░░░░│             │
        K_A train (R_A's cands)         │                 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│             │  from scratch, minutes
        gate vs (R_B', K')              │                 │                       │▒▒▒▒▒▒▒▒▒▒▒▒▒│  no regression
pass B  R_B = R_A + holdout             │                 │█████████████████████████████████████│  fine-tune, same recipe
        K_B (R_A's cands)               │                 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  R_A never saw these weeks
        serve (R_B, K_B); the next cycle warm-starts from R_A, never R_B
```

- **Pass A (evaluate and monitor).** R_A = R_A' warm-started on the delta `[N' − 6w, N − 6w)`
  plus a replay sample of older rows (no forgetting). K_A trains on R_A's candidates over
  `[N − 6w, N − 2w)`. The ranker is retrained from scratch each cycle (minutes on CPU); continuing
  K' with LightGBM's `init_model` would keep stacking trees fitted to old windows. Pass A's
  numbers are the cycle's quality record (`evaluation/cycles/<run>/`).
- **Gate (no regression).** On the test window, the pair (R_A, K_A) must stay within
  `MAX_REGRESSION` (default 0.002) of the serving champion (R_B', K') on warm recall@20 and NDCG@20,
  and R_A's retrieval recall@100 within the same margin of R_B''s. A refresh exists to stay fresh,
  so it needs "not worse", not "better". The champion is evaluated on the same window: R_B' was
  trained through N' ≈ N − 4w and K' on R_A''s candidates up to N', both before `N − 2w`. The
  champion has seen 2 more weeks than R_A, so the comparison leans conservative. Also reported:
  (R_A, K') (does the old ranker still fit the new candidates).
- **Pass B (refresh to serve).** R_B = R_A fine-tuned on the 6 held-out weeks `[N − 6w, N)` with
  the same recipe; K_B = a ranker trained on R_A's candidates over `[N − 6w, N)` (R_A never saw
  them, so no leakage; the test weeks are training data now). There is no unseen window left for
  R_B / K_B, so before serving: sanity checks (R_B's recall on its last training week not below
  R_A's on the same rows, top-20 churn of (R_B, K_B) vs (R_A, K_A) on a sample of users under
  `MAX_CHURN`, catalog coverage, embedding health). After serving: the delayed-label check; if
  live recall@20 drops by more than `MAX_LIVE_DROP` vs the previous run, the previous pair is
  restored automatically (`ROLLBACK=1`) and the next inference run rewrites the lists.
- **Records.** `pair.json` holds the serving pair (R_B, K_B) **and** its evaluation twin (R_A,
  K_A); `history.jsonl` gets one line per cycle. The current champion (cutoff 2026-05-01, never
  trained past it) is a valid R_A: the first cycle starts from it and produces the first R_B.

#### 3. A retriever challenger (new architecture or recipe, from scratch)

```
time ──────────────────────────────────────────────────────────────────────────►
                                           N − 6w              N − 2w        N
challenger R_C    ██████████████████████████│                                    from scratch
R_C valid                                   │░░░░░░░░░░░░░░░░░░│
K_C train (R_C's candidates)                │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
champion (R_B', K') trained through N'      ████████████████████████│ ≈ N − 4w
gates: (R_C, K_C), (R_C, K') vs (R_B', K')                     │▒▒▒▒▒▒▒▒▒▒▒▒▒│
```

- **Train**: everything before `N − 6w`, epochs picked on `[N − 6w, N − 2w)`. Not on the
  champion's validation rows: the champion R_B' has already trained through N'.
- **Retriever gate on the test window only**: R_B' has seen `[N − 6w, N')`, so the validation
  window can't compare the two; the test window is unseen by both. Warm recall@50 must beat R_B''s
  by `MIN_IMPROVEMENT` and recall@100 must not drop.
- **Rankers**: K_C on R_C's candidates over `[N − 6w, N − 2w)`, and the champion ranker K' run on
  R_C's candidates. The better of (R_C, K_C) and (R_C, K') must beat (R_B', K') on the test window
  by `MIN_IMPROVEMENT` (a challenger must win, unlike a refresh).
- **If promoted**: pass B as in 2 (R_C fine-tuned on `[N − 6w, N)` serves; the ranker trained on
  R_C's candidates over the whole holdout serves with it), and R_C becomes the evaluation lineage
  the next cycle warm-starts from.

#### 4. A ranker challenger (features, params; same retriever)

```
time ──────────────────────────────────────────────────────────────────────────►
                                           N − 6w              N − 2w        N
evaluation retriever R_A (latest cycle) ███│
K_new and K_champion train (R_A's candidates) │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
gate: (R_A, K_new) vs (R_A, K_champion)                         │▒▒▒▒▒▒▒▒▒▒▒▒▒│
```

- Candidates come from the evaluation retriever R_A of the latest cycle (never R_B: it saw every
  window). The cycle's N fixes the windows; a ranker experiment between cycles reuses them.
- Both rankers are trained the same way on the same window and compared on the same retriever's
  candidates, so the difference is the ranker alone. K_new must beat K_champion's pair recall@20
  by `MIN_IMPROVEMENT` (and pass the ranker gate vs retrieval order).
- **If promoted**: retrain K_new on R_A's candidates over `[N − 6w, N)` (as pass B does) and serve
  it with the current R_B. A feature-schema change goes out with phase 5's serving deploy first
  when the online path must read it.

**Settings** (phase 3 / 8): `HOLDOUT_WEEKS` (6), `TEST_WEEKS` (2), `MAX_REGRESSION` (0.002, refresh
gate), `MIN_IMPROVEMENT` (challengers), `REPLAY_FRACTION` (older rows mixed into a warm start),
`MAX_CHURN`, `MAX_LIVE_DROP`. `VALIDATION_FRACTION` stays for experiments on a fixed split.

### Delivery sequence

Each phase ships on its own (CI, docs, deploy), in this order:

1. **Ranker experiments** (`training/src/steam_training/ranker/`): groups from a retriever's
   candidates after its cutoff, the feature module, LightGBM `lambdarank`, the report (retrieval
   vs ranker order, drop-one-family deltas, history-length split), artifacts under
   `rankers/<id>/`, `python -m steam_training rank --retriever-id <id>` locally and as a CPU job.
   Popularity computed from `interactions`.
2. **ETL: popularity mart** (`game_popularity_weekly`); training, inference's `__popular__` and
   spec 15 read it.
3. **Pair evaluation and promotion** (no live MLflow): time-based windows (`HOLDOUT_WEEKS`,
   `TEST_WEEKS`, see "Time windows"), ranker gate, pair evaluation of a new retriever (both
   rankers), final-list gate, `pair.json`, `history.jsonl`, `ROLLBACK=1`, `training-rank.yml`
   and `training-promote.yml` changes, the numpy `ranker.npz` export.
4. **Batch inference**: the ranker in `RANKER_MODE=shadow`, then `on`; the delayed-label check.
5. **Online serving**: `game_features.npz` export, the numpy tree evaluator in the Lambda,
   parity test; deploy serving before promoting a ranker that serving must read.
6. **Language**: `language` in `interactions` (ETL), the language feature (a
   `FEATURE_SCHEMA_VERSION` bump and a ranker retrain), per-language popular lists.
7. **Training orchestration**: a training Step Function (retriever training job → ranker job →
   pair evaluation → manual approval → promote → inference run), started by hand or by
   EventBridge.
8. **Continuous training** (its own spec, following "Time windows" 2): the monthly pass A /
   pass B cycle with the evaluation and serving lineages, the no-regression gate, sanity checks
   and the delayed-label automatic rollback. Full retrains from scratch stay on the challenger
   path (3).

### Out of scope

Review text features, the neural ranker (see "Model"), batch-only features (negatives, history
beyond 20) as a second model.

Follow-ups that need an ETL change (the columns are scraped but dropped by `stg_steam__reviews`):

- **Reviewer language** (phase 6): see "Features" and "Cold start".
- **Playtime** (`playtime_at_review`): a playtime-weighted tag profile scores the same as the
  count-weighted one (AUC 0.649 vs 0.648), so not worth it as a feature; only graded labels (hours
  played on the target) remain untested.

### Tests / docs / infra

- Unit tests: group building (label, dropped groups, warm-only groups, per-user cap, no feature
  from t or later), each feature family on hand-built histories, features unchanged when the
  retriever's raw scores are rescaled (scale-free retrieval features), popularity as of the weeks
  before t, the ranker gate, artifact save / load, the numpy tree evaluator vs LightGBM.
- Unit tests, promotion: retriever-only promotion keeps the champion ranker when the
  compatibility check passes and drops it when it fails, ranker-only promotion, the final-list
  gate, `pair.json` / `history.jsonl`, `ROLLBACK=1` for each case.
- Inference: with and without a champion ranker, a ranker not in `pair.json` is ignored, shadow
  mode writes retrieval order, a ranker failure falls back per batch; with an attention champion;
  the serving contract test still passes. Serving: ranked and unranked online responses, a
  ranker / pair mismatch falls back to retrieval order.
- End to end on the synthetic marts (`conftest.py`): the ranker beats retrieval order on data
  with a signal the towers cannot see (e.g. a burst of recent popularity).
- `uv add lightgbm` in `training` and `inference` (images only; the serving Lambda gets the
  numpy evaluator, no LightGBM).
- New settings in `TrainingSettings` / `InferenceSettings` + SSM (`infrastructure/training.tf`,
  `infrastructure/inference.tf`); `training-rank.yml` (manual, like `training-cd.yml`).
- Re-check the feature families on the champion's real candidates (the EDA used a proxy
  retriever), including the cost of the decoupled retrieval features vs the raw score.
- `training/README.md`, `training/CLAUDE.md` (invariants: ranker features only from before t, no
  ranker feature depends on a retriever version, never serve a pair not evaluated together),
  `inference/README.md`, `inference/CLAUDE.md`, `docs/deployment.md` (adding a ranker to the
  current champion, rollback).
