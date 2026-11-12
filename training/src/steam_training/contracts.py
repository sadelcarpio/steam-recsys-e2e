"""Pydantic contracts of the training artifacts, the training -> inference / promotion interface.

S3 layout (bucket `model-artifacts-<acct>`):
    models/<model_id>/user_tower.pt, item_tower.pt   state dicts (load with `artifacts.load_model`)
    models/<model_id>/tracking.json                  TrackingRecord (experiment tracking, spec 18)
    models/<model_id>/metadata.json                  ModelMetadata
    evaluation/<model_id>/metrics.json               EvaluationReport (promotion job)
    models/champion/..., evaluation/champion/...     copies of the promoted model
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Reserved ids of every lkp_* vocabulary (see etl/src/steam_etl/contracts.py).
PADDING_ID = 0
OOV_ID = 1
FIRST_ID = 2
# Length of the mart's history (`games_reviewed_positive`, dbt var `user_history_length`): what
# inference and serving read today. Longer histories (A3) are derived in training from
# `interactions` (`data.HistoryIndex`) and set per model (`ModelConfig.history_length`).
USER_HISTORY_LENGTH = 5

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
