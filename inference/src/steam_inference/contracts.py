"""Pydantic contracts of the inference output (the inference -> serving interface) and of the
LLM reranking response. Serving reads both tables (serving/src/steam_serving/contracts.py):
`tests/test_serving_contract.py` checks that it parses every item written here.

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

The reserved item `user_id = "__popular__"` (model_id "popularity", never reranked) holds the
most reviewed-positive games of the recent reviews, the fallback for users without an item.
Its `score` is the game's share of the top game's positive reviews, in (0, 1].

DynamoDB table `game-details`, one item per catalog game, insert-only (details are static):
    game_id N partition key (Steam appid), name S, short_description S?, header_image S?,
    release_date S?, is_free BOOL?, price N?, developers / publishers / genres / categories L of S

Online bundle (`online.py`), for serving's POST /recommendations: the catalog side
s3://<model-artifacts>/serving/online/catalog.npz (the arrays below, rewritten every run) and
manifest.json (`OnlineBundleManifest`), pinned to that catalog's S3 version and naming the
model's numpy user tower, models/<model_id>/user_tower.npz (written by training, see
training/src/steam_training/export.py).

Search index (`search.py`), for the frontend: s3://<model-artifacts>/serving/search/games.json
(`SearchIndex`, gzipped), the same games as the online catalog, served by CloudFront.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

POPULAR_USER_ID = "__popular__"
POPULARITY_MODEL_ID = "popularity"

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


_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def clean_text(value: str | None) -> str | None:
    """Plain text from Steam's store text: tags dropped, HTML entities decoded, spaces collapsed."""
    if value is None:
        return None
    text = _SPACES.sub(" ", html.unescape(_TAG.sub(" ", value))).strip()
    return text or None


class GameDetails(_Frozen):
    """Human-readable details of a game (mart `game_details`), for serving / a frontend."""

    game_id: int
    name: str
    short_description: str | None = None
    header_image: str | None = None
    release_date: str | None = None
    is_free: bool | None = None
    price: Annotated[float, Field(ge=0)] | None = None
    developers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    def to_item(self) -> dict[str, Any]:
        """DynamoDB item (resource API types: Decimal numbers, no None values)."""
        item: dict[str, Any] = self.model_dump(exclude_none=True)
        if self.price is not None:
            item["price"] = Decimal(f"{self.price:.2f}")
        return item


# Arrays of the online catalog (numpy .npz): the current catalog scored by the model's item
# tower. Keep in sync with serving/src/steam_serving/online.py.
ONLINE_BUNDLE_FORMAT = 2
ONLINE_BUNDLE_ARRAYS = (
    "item_embeddings",  # float32 [rows, output_dim], L2-normalized item tower output
    "item_game_id",  # int64 [rows] Steam appid of each catalog row
    "item_game_idx",  # int64 [rows] dense id (lkp_games) of each catalog row
    "item_name_utf8",  # uint8 [bytes] every row's name, UTF-8, concatenated
    "item_name_offsets",  # int64 [rows + 1] row i's name is item_name_utf8[o[i]:o[i + 1]]
)


class OnlineBundleManifest(_Frozen):
    """serving/online/manifest.json: the catalog (pinned to its S3 version, so a reader never
    mixes a new manifest with an older catalog) and the user tower of the same model
    (models/<model_id>/ is immutable, so its key needs no version)."""

    format_version: int = ONLINE_BUNDLE_FORMAT
    model_id: str
    generated_at: datetime
    catalog_key: str
    catalog_version_id: str | None = None  # None: unversioned storage (local dry runs)
    catalog_sha256: str
    catalog_games: int = Field(ge=1)
    user_tower_key: str  # models/<model_id>/user_tower.npz, same bucket


SEARCH_INDEX_FORMAT = 1


class SearchIndex(_Frozen):
    """serving/search/games.json: the frontend's game search. Keep in sync with
    frontend/src/api.ts (`SearchIndex`)."""

    format_version: int = SEARCH_INDEX_FORMAT
    model_id: str
    generated_at: datetime
    # [appid, name, reviews], most reviewed first
    games: list[tuple[int, str, Annotated[int, Field(ge=0)]]]


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
    popular_games: int = 0
    game_details_written: int = 0
    online_bundle: str | None = None
    search_index: str | None = None
    snapshots: dict[str, int | None] = Field(default_factory=dict)
    seconds: float = 0.0
