"""Pydantic contracts of the training artifacts, the training -> inference / promotion interface.

S3 layout (bucket `model-artifacts-<acct>`):
    models/<model_id>/user_tower.pt, item_tower.pt   state dicts (load with `artifacts.load_model`)
    models/<model_id>/tracking.json                  TrackingRecord (experiment tracking, spec 18)
    models/<model_id>/metadata.json                  ModelMetadata
    evaluation/<model_id>/metrics.json               EvaluationReport (promotion job)
    models/champion/..., evaluation/champion/...     copies of the promoted model
    rankers/<ranker_id>/model.txt                    LightGBM text dump of a ranker (spec 9)
    rankers/<ranker_id>/metadata.json                RankerMetadata
    evaluation/rankers/<ranker_id>/ranker.json       RankerReport (`python -m steam_training rank`)
    rankers/champion/...                             copy of the serving ranker (none: deleted)
    evaluation/pairs/<retriever>__<ranker>.json      PairReport (`python -m steam_training pair`)
    evaluation/champion/pair.json                    PairRecord: the serving (retriever, ranker)
    evaluation/champion/history.jsonl                one PairRecord per promotion / rollback
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Reserved ids of every lkp_* vocabulary (see etl/src/steam_etl/contracts.py).
PADDING_ID = 0
OOV_ID = 1
FIRST_ID = 2
# Length of the mart's history (`games_reviewed_positive`, dbt var `user_history_length`): what
# inference and serving read today. Longer histories (A3) are derived in training from
# `interactions` (`data.HistoryIndex`) and set per model (`ModelConfig.history_length`).
USER_HISTORY_LENGTH = 5
# A7: Steam API review languages; id = index + FIRST_ID (0 padding, 1 = OOV: unknown / none).
# Append only: the ids are part of a trained model.
STEAM_LANGUAGES = (
    "english", "schinese", "tchinese", "japanese", "koreana", "thai", "bulgarian", "czech",
    "danish", "german", "spanish", "latam", "greek", "french", "italian", "indonesian",
    "hungarian", "dutch", "norwegian", "polish", "portuguese", "brazilian", "romanian",
    "russian", "finnish", "swedish", "turkish", "vietnamese", "ukrainian",
)  # fmt: skip
NUM_LANGUAGE_IDS = FIRST_ID + len(STEAM_LANGUAGES)

# Bump when the model layout changes so an older champion is not loaded into new code.
ARCHITECTURE_VERSION = 1

Recall = Annotated[float, Field(ge=0, le=1)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VocabSizes(_Frozen):
    """max(id) + 1 of each lookup at training time. Larger ids map to OOV."""

    games: int = Field(ge=FIRST_ID)
    developers: int = Field(ge=FIRST_ID)
    publishers: int = Field(ge=FIRST_ID)
    genres: int = Field(ge=FIRST_ID)
    categories: int = Field(ge=FIRST_ID)
    # Steam user tags (spec 8); None for a model trained without them (no tags module).
    tags: int | None = Field(None, ge=FIRST_ID)


# history time gaps (rejected, code removed): unset in the models saved while it existed
_RETIRED_FIELDS = ("history_gap_seconds",)


class ModelConfig(_Frozen):
    architecture_version: int = ARCHITECTURE_VERSION
    vocab: VocabSizes
    game_embedding_dim: int
    attribute_embedding_dim: int
    hidden_dim: int
    output_dim: int
    temperature: float
    # A4 (spec 17): tag profile of the history in the user tower; False for older models
    user_tag_profile: bool = False
    # A3 (spec 17): games in the user tower's history (most recent first), and how they are
    # pooled: "mean" (the masked mean) or "attention" (also a pre-LN self-attention block over
    # the history + a learned summary token). Defaults = every model before A3.
    history_length: int = Field(USER_HISTORY_LENGTH, ge=1, le=512)
    history_encoder: Literal["mean", "attention"] = "mean"
    attention_heads: int = Field(2, ge=1)
    attention_layers: int = Field(1, ge=1)
    # A6 (spec 17): the item tower also reads recency features as of the review / catalog day
    # (log1p reviews in the last 30 / 90 days, log1p days since release, release known);
    # False for older models
    item_recency: bool = False
    # A7 (spec 17): the user tower also reads the language of the user's previous review
    # (an embedding of STEAM_LANGUAGES, OOV when unknown); False for older models
    user_language: bool = False

    @model_validator(mode="before")
    @classmethod
    def _drop_retired(cls, data: Any) -> Any:
        """Experiment fields whose code was removed (rejected arms) are accepted while unset,
        so the saved models that carry them (e.g. the shared A3 controls) still load."""
        if isinstance(data, dict) and any(data.get(k) is not None for k in _RETIRED_FIELDS):
            raise ValueError(f"model uses a retired experiment input: {_RETIRED_FIELDS}")
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in _RETIRED_FIELDS}
        return data


class SplitInfo(_Frozen):
    """Temporal split: the model trains on timestamp < cutoff and is evaluated on
    cutoff <= timestamp < end (`end` None = no upper bound). `evaluation` names that window:
    `validation` (FIT_ON=train; a held-out test window, if any, starts at `end` and is never
    read) or `test` (FIT_ON=train+validation: trained through the validation window)."""

    cutoff: datetime
    train_rows: int = Field(ge=0)
    # rows of the evaluation window (named validation_rows for older metadata)
    validation_rows: int = Field(ge=0)
    end: datetime | None = None
    evaluation: Literal["validation", "test"] = "validation"


class SegmentRecall(_Frozen):
    rows: int = Field(ge=0)
    recall: dict[int, Recall]


class RecallMetrics(_Frozen):
    """Recall@K on the positive validation rows. `warm` (non-empty history) is what inference
    serves and what promotion compares; `cold` rows share one "no history" query."""

    warm: SegmentRecall
    cold: SegmentRecall
    all: SegmentRecall


class ModelMetadata(_Frozen):
    model_id: str
    created_at: datetime
    config: ModelConfig
    split: SplitInfo
    # Iceberg snapshot ids read (table name -> snapshot id) to reproduce the training data.
    snapshots: dict[str, int | None]
    hyperparameters: dict[str, object]
    validation: RecallMetrics
    popularity_baseline: RecallMetrics
    epoch_losses: list[float]
    # epoch whose weights were saved (best monitor score, KEEP_BEST_EPOCH); None in older models
    best_epoch: int | None = None


class ModelEvaluation(_Frozen):
    model_id: str
    metrics: RecallMetrics


class EvaluationReport(_Frozen):
    """evaluation/<model_id>/metrics.json: candidate and champion on the same validation rows
    (after both models' training cutoffs, so neither saw them)."""

    model_id: str
    evaluated_at: datetime
    primary_k: int
    split: SplitInfo
    candidate: ModelEvaluation
    champion: ModelEvaluation | None
    popularity_baseline: RecallMetrics
    promoted: bool
    reason: str

    def primary(self, metrics: RecallMetrics) -> float:
        return metrics.warm.recall[self.primary_k]


# ---- experiment tracking (spec 18) ----


class ActivityBucket(_Frozen):
    """Recall of the validation rows whose user had `min_activity`..`max_activity` reviews
    before the cutoff (`new`: none). Buckets are percentiles of the validation users."""

    name: str
    min_activity: int = Field(ge=0)
    max_activity: int = Field(ge=0)
    rows: int = Field(ge=0)
    users: int = Field(ge=0)
    recall: dict[int, Recall]


class Concentration(_Frozen):
    """How concentrated a model's top-k lists are over the current catalog (every game's newest
    features; A6 recency as of the newest week), for a sample of recent users: what a deployed
    model would serve now, weeks past its cutoff. A model that sends one game to most users
    (e.g. an unseen new release scored from content + recency alone) shows a high `reach`."""

    k: int
    lists: int
    reach: float  # share of the lists containing the most common game
    top_game_idx: int  # that game
    top10_share: float  # share of all slots taken by the 10 most common games
    unseen_share: float  # share of slots with games the model never trained on (OOV)
    distinct: int  # distinct games over all lists


class RetrievalDiagnostics(_Frozen):
    """Every offline metric of one model (or the popularity baseline) on the validation rows."""

    recall: RecallMetrics
    # users with at least one validation target in their top K (per row, recall = hit rate)
    user_hit_rate: dict[int, Recall]
    # mean over users of their rows' recall (each user weighs the same)
    user_recall: dict[int, Recall]
    users: int = Field(ge=0)
    by_activity: list[ActivityBucket]
    # share of the catalog retrieved in the top K of at least one row
    coverage: dict[int, Recall]
    # Gini of how often each catalog game is retrieved in the top max(K) (1 = all on one game)
    exposure_gini: float = Field(ge=0, le=1)


class EmbeddingHealth(_Frozen):
    """Collapse checks: random vectors have mean cosine ~0 and an effective rank near `dim`;
    collapsed ones have mean cosine near 1 and an effective rank near 1."""

    rows: int = Field(ge=0)
    dim: int = Field(ge=1)
    mean_cosine: float
    effective_rank: float = Field(ge=0)
    effective_rank_ratio: float = Field(ge=0, le=1)


class EpochMetrics(_Frozen):
    epoch: int = Field(ge=1)
    loss: float
    monitor_warm_recall: float | None = None  # at primary_k, EPOCH_EVAL_ROWS rows
    seconds: float | None = None  # None for epochs restored from a checkpoint


class TrackingRecord(_Frozen):
    """models/<model_id>/tracking.json: one training run, everything experiment tracking needs.
    Live runs mirror it into MLflow while training; `python -m steam_training track` replays it
    (SageMaker jobs, or runs without a tracking server)."""

    model_id: str
    started_at: datetime
    ended_at: datetime
    primary_k: int
    # which window `model` / `popularity` were measured on (MLflow prefix `validation/`, `test/`)
    evaluation: Literal["validation", "test"] = "validation"
    # lineage (git commit, snapshot ids, cutoff, feature signature, ...)
    tags: dict[str, str]
    params: dict[str, str]
    epochs: list[EpochMetrics]
    model: RetrievalDiagnostics
    popularity: RetrievalDiagnostics
    item_embeddings: EmbeddingHealth
    user_embeddings: EmbeddingHealth
    # A6 runs: the weekly recent-popularity baseline (`evaluation.recent_popularity_scores`)
    recent_popularity: RetrievalDiagnostics | None = None
    # exposure over the current catalog (`evaluation.concentration`); None for older records
    concentration: Concentration | None = None


# ---- ranker (spec 9) ----


class RankingMetrics(_Frozen):
    """Ranking quality of one ordering of the test groups (one held-out game per group).
    `recall[k]` = share of groups with the game in the top k (precision@k = recall@k / k);
    groups whose game the retriever missed count as misses."""

    groups: int = Field(ge=0)
    recall: dict[int, Recall]
    ndcg: dict[int, Recall]
    # mean of 1 / position (1-based) when the game is in the top k, else 0: average precision@k
    # with one relevant game
    mrr: dict[int, Recall] = Field(default_factory=dict)


class RankerWindow(_Frozen):
    """Time split of a ranker: its candidates come from a retriever trained before
    `retriever_cutoff` (T1); groups in [T1, t2) train it (the last share by time stops early),
    groups in [t2, end] test it."""

    retriever_cutoff: datetime
    t2: datetime
    end: datetime
    train_groups: int = Field(ge=0)
    early_stopping_groups: int = Field(ge=0)
    test_groups: int = Field(ge=0)


class RankerComparison(_Frozen):
    """Retrieval order vs ranker order on the same test groups."""

    retrieval: RankingMetrics
    ranker: RankingMetrics


class ListMetrics(_Frozen):
    """Per-user list metrics: one list per (user, snapshot T) built from the history before T,
    relevant = the games the user reviews positively in [T, T + horizon) (not reviewed before
    T). Averaged over (user, snapshot) pairs, so every user weighs the same at each snapshot."""

    lists: int = Field(ge=0)
    # hits in the top k / k
    precision: dict[int, Recall]
    # hits in the top k / min(k, relevant games)
    recall: dict[int, Recall]
    # average precision@k: mean over the hits of precision at their position, / min(k, relevant)
    map: dict[int, Recall]


class UserListComparison(_Frozen):
    """Retrieval order vs ranker order of the same per-user lists (the batch path's view: one
    list written per run, judged against what the user does until the next run)."""

    snapshots: list[datetime]
    horizon_days: int = Field(ge=1)
    retrieval: ListMetrics
    ranker: ListMetrics


class RankerMetadata(_Frozen):
    ranker_id: str
    created_at: datetime
    # the retriever whose candidates trained it (candidates and windows depend on it; no
    # feature does)
    retriever_id: str
    feature_schema_version: int
    feature_names: list[str]
    candidates: int = Field(ge=1)
    top_k: int = Field(ge=1)
    history_length: int = Field(ge=1)
    window: RankerWindow
    snapshots: dict[str, int | None]
    hyperparameters: dict[str, object]
    best_iteration: int = Field(ge=0)
    test: RankerComparison


class RankerReport(_Frozen):
    """evaluation/rankers/<ranker_id>/ranker.json: the ranker on its test window."""

    ranker_id: str
    retriever_id: str
    evaluated_at: datetime
    top_k: int
    window: RankerWindow
    # share of test groups whose game is among the retriever's candidates (the ceiling)
    candidate_recall: Recall
    test: RankerComparison
    # the same, only the test groups whose game is among the candidates (the ranker's own skill)
    conditional: RankerComparison | None = None
    # per-user lists at weekly snapshots of the test window (None: not computed)
    user_lists: UserListComparison | None = None
    # the same, split by the length of the user's history (short: < history_length games)
    by_history: dict[str, RankerComparison]
    # share of the total split gain per feature
    importances: dict[str, float]
    # recall@top_k change when a feature family is left out (retrained without it)
    ablation: dict[str, float]
    gate_passed: bool
    reason: str


# ---- serving pair (spec 9, phase 3) ----


class PairEvaluation(_Frozen):
    """One (retriever, ranker) on the common test groups. `final` = the list the pair serves
    (ranker order, or retrieval order without a ranker); `retrieval` = the retriever's own order
    (its recall@K is the ranker's ceiling)."""

    retriever_id: str
    ranker_id: str | None
    final: RankingMetrics
    retrieval: RankingMetrics
    user_lists: ListMetrics | None = None
    # the retriever's top-k over the current catalog (gated by PAIR_MAX_GAME_REACH)
    concentration: Concentration | None = None


class PairReport(_Frozen):
    """evaluation/pairs/<retriever>__<ranker>.json: a promotion decision between the champion
    pair and challenger pairs, all on groups at or after `window_start` (no compared model
    trained on them)."""

    evaluated_at: datetime
    window_start: datetime
    top_k: int
    candidates: int
    champion: PairEvaluation | None
    challengers: list[PairEvaluation]
    chosen: PairEvaluation | None
    promoted: bool
    reason: str
    # evaluated only (PAIR_DRY_RUN): `chosen` is the pair that would be promoted, applied later
    # by APPLY_REPORT (the training Step Function's approval sits in between)
    dry_run: bool = False


class PairRecord(_Frozen):
    """evaluation/champion/pair.json (and one line of history.jsonl): the pair being served."""

    retriever_id: str
    ranker_id: str | None
    promoted_at: datetime
    reason: str
    window_start: datetime | None = None
    final: RankingMetrics | None = None
