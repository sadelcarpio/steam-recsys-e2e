"""Pydantic contracts of the inference output (the inference -> serving interface) and of the
LLM reranking response.

DynamoDB table `game-explainable-recommendations`, one item per user. A run only (re)writes
the users whose `content_hash` changed and deletes the users that are gone:
    user_id          S   partition key (Steam 64-bit id as a string: exceeds JS safe integers)
    recommendations  L   of M, best first: {game_id N, name S, score N, explanation S (top N only)}
    model_id         S   model that retrieved the candidates
    generated_at     S   ISO-8601 UTC time of the run
    reranked         BOOL  true when the LLM reordered the list (and wrote the explanations)
    rerank_model     S   Bedrock model id (reranked items only)
    content_hash     S   hash of what the user sees (`UserRecommendations.content_hash`)

`score` and `generated_at` are as of the last write: an unchanged list is not rewritten, even
when its scores moved a little (weekly `reviews_ratio` updates shift every score).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# Score = cosine similarity of the user and game embeddings (before any reranking).
Score = Annotated[float, Field(ge=-1.0001, le=1.0001)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Recommendation(_Frozen):
    game_id: int  # Steam appid
    name: str
    score: Score
    explanation: str | None = None


class UserRecommendations(_Frozen):
    user_id: str
    recommendations: list[Recommendation] = Field(min_length=1)
    model_id: str
    generated_at: datetime
    reranked: bool
    rerank_model: str | None = None

    def content_hash(self) -> str:
        """Hash of the visible content: model, rerank flag / model, and the ordered games with
        their names and explanations. Scores and the generation time are left out."""
        content = {
            "model_id": self.model_id,
            "reranked": self.reranked,
            "rerank_model": self.rerank_model,
            "recommendations": [[r.game_id, r.name, r.explanation] for r in self.recommendations],
        }
        payload = json.dumps(content, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_item(self) -> dict[str, Any]:
        """DynamoDB item (resource API types: Decimal numbers, no None values)."""
        item: dict[str, Any] = {
            "user_id": self.user_id,
            "recommendations": [
                {
                    "game_id": r.game_id,
                    "name": r.name,
                    "score": Decimal(f"{r.score:.4f}"),
                    **({"explanation": r.explanation} if r.explanation else {}),
                }
                for r in self.recommendations
            ],
            "model_id": self.model_id,
            "generated_at": self.generated_at.isoformat(),
            "reranked": self.reranked,
            "content_hash": self.content_hash(),
        }
        if self.rerank_model:
            item["rerank_model"] = self.rerank_model
        return item


class RankedCandidate(BaseModel):
    """One entry of the LLM's ranking: a candidate number from the prompt (1-based)."""

    model_config = ConfigDict(extra="ignore")

    candidate: int
    explanation: str | None = None


class LlmRanking(BaseModel):
    """Input of the `submit_ranking` tool the LLM must call."""

    model_config = ConfigDict(extra="ignore")

    ranking: list[RankedCandidate]


class InferenceSummary(_Frozen):
    model_id: str | None
    skipped: bool
    reason: str = ""
    users: int = 0
    catalog_games: int = 0
    reranked_users: int = 0
    rerank_failures: int = 0
    written: int = 0
    unchanged: int = 0
    deleted: int = 0
    snapshots: dict[str, int | None] = Field(default_factory=dict)
    seconds: float = 0.0
