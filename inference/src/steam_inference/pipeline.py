"""Batch inference: champion model -> top K per user -> LLM rerank (top reviewers) -> DynamoDB
(only the users whose recommendations changed are written)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

import numpy as np
from steam_training.artifacts import ArtifactStore
from steam_training.contracts import ARCHITECTURE_VERSION, PADDING_ID
from steam_training.data import TableSource

from steam_inference.config import InferenceSettings
from steam_inference.contracts import InferenceSummary, Recommendation, UserRecommendations
from steam_inference.features import InferenceData, load_inference_data
from steam_inference.rerank import RankFn, Reranked, RerankRequest, rerank_all
from steam_inference.retrieval import Candidates, retrieve
from steam_inference.writer import Writer

log = logging.getLogger(__name__)


def run_inference(
    settings: InferenceSettings,
    source: TableSource,
    store: ArtifactStore,
    writer: Writer,
    rank_fn: RankFn | None,
    *,
    now: datetime | None = None,
) -> InferenceSummary:
    started = time.monotonic()
    now = now or datetime.now(UTC)
    metadata = store.load_metadata(settings.model_id)
    if metadata is None:
        reason = f"no model at s3://{store.bucket}/models/{settings.model_id}/: skipped"
        log.warning(reason)
        return InferenceSummary(model_id=None, skipped=True, reason=reason)
    if metadata.config.architecture_version != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"model {metadata.model_id} has architecture v{metadata.config.architecture_version}, "
            f"this image expects v{ARCHITECTURE_VERSION}: deploy a matching inference image"
        )
    model, metadata = store.load_model(settings.model_id)
    log.info("serving model %s (trained %s)", metadata.model_id, metadata.created_at)

    data = load_inference_data(source, max_users=settings.max_users)
    candidates = retrieve(
        model,
        data.games,
        data.users,
        data.reviews,
        k=settings.top_k,
        user_batch_size=settings.user_batch_size,
        item_batch_size=settings.item_batch_size,
    )

    reranked: dict[int, Reranked] = {}
    failures = 0
    if rank_fn is not None:
        requests = {
            user: rerank_request(data, candidates, user)
            for user in select_rerank_users(data, candidates, settings)
        }
        log.info("reranking %d users with %s", len(requests), settings.bedrock_model_id)
        reranked, failures = rerank_all(
            rank_fn,
            requests,
            explain_top_n=settings.explain_top_n,
            concurrency=settings.rerank_concurrency,
        )

    stored = writer.stored_hashes()
    changes = ChangedOnly(
        user_recommendations(
            data,
            candidates,
            reranked,
            model_id=metadata.model_id,
            rerank_model=settings.bedrock_model_id,
            now=now,
        ),
        stored,
    )
    written = writer.write(changes)
    deleted = 0
    gone = stored.keys() - changes.seen
    if gone and settings.max_users:
        log.info("partial run (MAX_USERS=%d): %d stored users kept", settings.max_users, len(gone))
    elif gone:
        deleted = writer.delete(sorted(gone))
    summary = InferenceSummary(
        model_id=metadata.model_id,
        skipped=False,
        users=len(data.users),
        catalog_games=len(data.games),
        reranked_users=len(reranked),
        rerank_failures=failures,
        written=written,
        unchanged=changes.unchanged,
        deleted=deleted,
        snapshots=data.snapshots,
        seconds=round(time.monotonic() - started, 1),
    )
    log.info("inference done: %s", summary.model_dump_json())
    return summary


class ChangedOnly:
    """Passes through the items whose content hash differs from the stored one, counting the
    unchanged ones and remembering every user id seen (the others are gone)."""

    def __init__(self, items: Iterable[UserRecommendations], stored: dict[str, str]) -> None:
        self.items = items
        self.stored = stored
        self.seen: set[str] = set()
        self.unchanged = 0

    def __iter__(self) -> Iterator[UserRecommendations]:
        for item in self.items:
            self.seen.add(item.user_id)
            if self.stored.get(item.user_id) == item.content_hash():
                self.unchanged += 1
            else:
                yield item


def select_rerank_users(
    data: InferenceData, candidates: Candidates, settings: InferenceSettings
) -> np.ndarray:
    """Positions of the users to rerank: >= RERANK_MIN_REVIEWS reviews and at least one
    candidate, the most active first (ties: lowest user id), at most RERANK_MAX_USERS."""
    users = data.users
    eligible = (users.review_count >= settings.rerank_min_reviews) & (candidates.rows[:, 0] >= 0)
    positions = np.flatnonzero(eligible)
    order = np.lexsort((users.user_id[positions], -users.review_count[positions]))
    return positions[order][: settings.rerank_max_users]


def rerank_request(data: InferenceData, candidates: Candidates, user: int) -> RerankRequest:
    """The user's `games_reviewed_positive` (what the user tower saw) + their candidates."""
    games = data.games
    liked = data.users.history[user]
    liked_rows = games.catalog.rows(liked[liked != PADDING_ID])
    rows, _ = candidates.of(user)
    return RerankRequest(
        liked=[games.describe(int(row)) for row in liked_rows if row >= 0],
        candidates=[games.describe(int(row)) for row in rows],
    )


def user_recommendations(
    data: InferenceData,
    candidates: Candidates,
    reranked: dict[int, Reranked],
    *,
    model_id: str,
    rerank_model: str,
    now: datetime,
) -> Iterator[UserRecommendations]:
    games = data.games
    for user in range(len(data.users)):
        rows, scores = candidates.of(user)
        if len(rows) == 0:
            continue
        result = reranked.get(user)
        order = result.order if result else range(len(rows))
        explanations = result.explanations if result else {}
        yield UserRecommendations(
            user_id=str(data.users.user_id[user]),
            recommendations=[
                Recommendation(
                    game_id=int(games.game_id[rows[p]]),
                    name=str(games.name[rows[p]]),
                    score=float(scores[p]),
                    explanation=explanations.get(p),
                )
                for p in order
            ],
            model_id=model_id,
            generated_at=now,
            reranked=result is not None,
            rerank_model=rerank_model if result else None,
        )
