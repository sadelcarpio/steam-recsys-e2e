"""Pydantic contracts of the training artifacts, the training -> inference / promotion interface.

S3 layout (bucket `model-artifacts-<acct>`):
    models/<model_id>/user_tower.pt, item_tower.pt   state dicts (load with `artifacts.load_model`)
    models/<model_id>/metadata.json                  ModelMetadata
    evaluation/<model_id>/metrics.json               EvaluationReport (promotion job)
    models/champion/..., evaluation/champion/...     copies of the promoted model
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Reserved ids of every lkp_* vocabulary (see etl/src/steam_etl/contracts.py).
PADDING_ID = 0
OOV_ID = 1
FIRST_ID = 2
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


class ModelConfig(_Frozen):
    architecture_version: int = ARCHITECTURE_VERSION
    vocab: VocabSizes
    game_embedding_dim: int
    attribute_embedding_dim: int
    hidden_dim: int
    output_dim: int
    temperature: float


class SplitInfo(_Frozen):
    """Temporal split: train = timestamp < cutoff, validation = timestamp >= cutoff."""

    cutoff: datetime
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)


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
