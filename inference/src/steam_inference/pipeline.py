"""Batch inference: champion model -> top K per user -> LLM rerank (top reviewers) -> DynamoDB
(only the users whose recommendations changed are written), plus the popularity fallback item
the details of new games (insert-only), the online catalog (item embeddings of the current
catalog, for the serving Lambda's POST /recommendations) and the demo user index (spec 13)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from itertools import chain

import numpy as np
from steam_training.artifacts import ArtifactStore
from steam_training.contracts import ARCHITECTURE_VERSION, PADDING_ID, USER_HISTORY_LENGTH
from steam_training.data import TableSource

from steam_inference.adult import adult_mask
from steam_inference.config import InferenceSettings
from steam_inference.contracts import (
    POPULAR_USER_ID,
    POPULARITY_MODEL_ID,
    InferenceSummary,
    LikedGame,
    Recommendation,
    UserRecommendations,
)
from steam_inference.details import sync_game_details
from steam_inference.features import InferenceData, load_inference_data
from steam_inference.online import BundleStore, build_catalog, encode_catalog
from steam_inference.online_check import check, load_sample, sample_lists, save
from steam_inference.ranking import (
    RankerStats,
    apply_ranker,
    cut,
    load_serving_ranker,
    online_ranker,
)
from steam_inference.rerank import (
    Reranked,
    RerankLlm,
    RerankOptions,
    RerankRequest,
    RerankStats,
    rerank_all,
)
from steam_inference.retrieval import Candidates, retrieve
from steam_inference.search import build_search_index, encode_search_index
from steam_inference.user_index import encode_user_index, load_user_index
from steam_inference.writer import Writer

log = logging.getLogger(__name__)
# Liked games in the rerank prompt and the stored `liked` list (most recent first): the mart's
# 5 whatever the model's history length (A3), so stored items don't change with the model.
LIKED_GAMES = USER_HISTORY_LENGTH


def run_inference(
    settings: InferenceSettings,
    source: TableSource,
    store: ArtifactStore,
    writer: Writer,
    llm: RerankLlm | None,
    *,
    details_writer: Writer | None = None,
    bundle_store: BundleStore | None = None,
    now: datetime | None = None,
    save_online_lists: bool = True,
) -> InferenceSummary:
    """`save_online_lists`: run the delayed-label check (reads and writes
    evaluation/online/ in the model bucket); off for dry runs."""
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

    data = load_inference_data(
        source,
        max_users=settings.max_users,
        popular_window_days=settings.popular_window_days,
        description_chars=settings.rerank_description_chars if settings.rerank_enabled else 0,
        require_tags=metadata.config.vocab.tags is not None,
        history_length=metadata.config.history_length,
        with_recency=metadata.config.item_recency,
        with_language=metadata.config.user_language,
    )
    # adult and unreleased (spec 10) games: never recommended, published or searchable
    adult = adult_mask(data.games) if settings.exclude_adult else np.zeros(len(data.games), bool)
    unreleased = (
        data.games.coming_soon
        if data.games.coming_soon is not None
        else np.zeros(len(data.games), bool)
    )
    excluded = adult | unreleased
    log.info(
        "excluding %d games of %d (%d adult, %d unreleased)",
        int(excluded.sum()),
        len(data.games),
        int(adult.sum()),
        int(unreleased.sum()),
    )
    # spec 9: the serving pair's ranker re-orders RETRIEVAL_CANDIDATES (shadow: compared only)
    ranker, ranker_reason = None, "RANKER_MODE=off"
    if settings.ranker_mode != "off":
        ranker, ranker_reason = load_serving_ranker(
            source,
            store,
            data.games.catalog,
            metadata.model_id,
            as_of=now,
            history_length=data.users.history.shape[1],
        )
    log.info("ranker: %s", ranker_reason)
    retrieved = retrieve(
        model,
        data.games,
        data.users,
        data.reviews,
        k=settings.retrieval_candidates if ranker else settings.top_k,
        user_batch_size=settings.user_batch_size,
        item_batch_size=settings.item_batch_size,
        excluded=excluded,
    )
    ranker_stats = RankerStats()
    if ranker is None:
        candidates = alternative = cut(retrieved, settings.top_k)
    else:
        ranked, ranker_stats = apply_ranker(
            ranker,
            data,
            retrieved,
            as_of=now,
            top_k=settings.ranker_top_k,
            chunk=settings.ranker_chunk_users,
        )
        if settings.ranker_mode == "on":
            candidates = cut(ranked, settings.ranker_top_k)
            alternative = cut(retrieved, settings.ranker_top_k)
        else:  # shadow: retrieval order is written, the ranker's lists are only compared
            candidates = cut(retrieved, settings.top_k)
            alternative = cut(ranked, settings.ranker_top_k)
    online_check = None
    if settings.online_check_users and save_online_lists:
        previous = load_sample(store.s3, store.bucket, settings.online_check_prefix)
        online_check = check(previous, data, now) if previous is not None else None
        save(
            store.s3,
            store.bucket,
            settings.online_check_prefix,
            sample_lists(
                data,
                candidates,
                alternative,
                users=settings.online_check_users,
                run_at=now,
                mode=settings.ranker_mode if ranker else "off",
                ranker_id=ranker.ranker_id if ranker else None,
            ),
            online_check,
        )

    online_bundle = search_index = None
    ranker_payload = None
    if bundle_store is not None:
        if store.has_user_tower_numpy(metadata.model_id):
            # spec 9: the same ranker and mode for POST /recommendations (a failed export
            # publishes the catalog without it: retrieval order online)
            if ranker is not None:
                try:
                    ranker_payload = online_ranker(
                        ranker,
                        data.games.catalog.items.game_idx[~excluded],
                        retriever_id=metadata.model_id,
                        candidates=settings.retrieval_candidates,
                        mode=settings.ranker_mode,
                        as_of=now,
                    )
                except Exception:
                    log.exception("online ranker export failed: catalog published without it")
            online_bundle = bundle_store.publish(
                encode_catalog(
                    build_catalog(
                        data.games,
                        candidates.item_embeddings,
                        keep=~excluded,
                    )
                ),
                model_id=metadata.model_id,
                generated_at=now,
                catalog_games=len(data.games) - int(excluded.sum()),
                user_tower_key=store.user_tower_numpy_key(metadata.model_id),
                ranker=ranker_payload,
            )
            # the frontend's search covers exactly the games the online model can use
            search_index = bundle_store.publish_search_index(
                encode_search_index(
                    build_search_index(data, model_id=metadata.model_id, now=now, excluded=excluded)
                )
            )
        else:
            log.warning(
                "model %s has no numpy user tower (saved before the export existed): online "
                "catalog not published; run `python -m steam_training export --model-id %s`",
                metadata.model_id,
                metadata.model_id,
            )

    reranked: dict[int, Reranked] = {}
    stats = RerankStats()
    if llm is not None:
        requests = {
            user: rerank_request(data, candidates, user, excluded)
            for user in select_rerank_users(data, candidates, settings)
        }
        log.info(
            "reranking %d users with %s (%s)",
            len(requests),
            settings.bedrock_model_id,
            settings.rerank_enabled_stages,
        )
        reranked, stats = rerank_all(
            llm,
            requests,
            RerankOptions(
                explain_top_n=settings.explain_top_n,
                stages=settings.rerank_enabled_stages,
                blend_weight=settings.rerank_blend_weight,
                shuffle=settings.rerank_shuffle,
            ),
            concurrency=settings.rerank_concurrency,
        )

    popular = popular_recommendations(data, k=settings.top_k, now=now, excluded=excluded)
    stored = writer.stored_hashes()
    changes = ChangedOnly(
        chain(
            user_recommendations(
                data,
                candidates,
                reranked,
                model_id=metadata.model_id,
                rerank_model=settings.bedrock_model_id,
                now=now,
                excluded=excluded,
            ),
            [popular] if popular else [],
        ),
        stored,
    )
    written = writer.write(changes)
    deleted = 0
    gone = stored.keys() - changes.seen
    if gone and settings.max_users and not settings.prune_unseen:
        log.info("partial run (MAX_USERS=%d): %d stored users kept", settings.max_users, len(gone))
    elif gone:
        deleted = writer.delete(sorted(gone))
    details_written, details_snapshot = 0, None
    details_deleted = 0
    if details_writer is not None:
        details_written, details_deleted, details_snapshot = sync_game_details(
            source, details_writer
        )
    user_index = None
    if bundle_store is not None:
        user_ids = load_user_index(source)
        if user_ids is not None:
            user_index = bundle_store.publish_user_index(
                encode_user_index(user_ids), max_user=len(user_ids), generated_at=now
            )
    summary = InferenceSummary(
        model_id=metadata.model_id,
        skipped=False,
        users=len(data.users),
        catalog_games=len(data.games),
        adult_games=int(adult.sum()),
        unreleased_games=int(unreleased.sum()),
        reranked_users=len(reranked),
        rerank_failures=stats.failures,
        rank_retries=stats.rank_retries,
        rank_fallbacks=stats.rank_fallbacks,
        explain_retries=stats.explain_retries,
        template_explanations=stats.templates,
        written=written,
        unchanged=changes.unchanged,
        deleted=deleted,
        popular_games=len(popular.recommendations) if popular else 0,
        game_details_written=details_written,
        game_details_deleted=details_deleted,
        online_bundle=online_bundle,
        online_ranker_id=ranker_payload.ranker_id if ranker_payload else None,
        search_index=search_index,
        user_index=user_index,
        snapshots={
            **data.snapshots,
            "game_details": details_snapshot,
            **(ranker.snapshots if ranker else {}),
        },
        ranker_id=ranker.ranker_id if ranker else None,
        ranker_mode=settings.ranker_mode if ranker else "off",
        ranker_reason=ranker_reason,
        ranked_users=ranker_stats.ranked_users,
        ranker_failures=ranker_stats.failures,
        ranker_overlap=ranker_stats.overlap,
        ranker_seconds=ranker_stats.seconds,
        online_check=online_check,
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


def popular_recommendations(
    data: InferenceData, *, k: int, now: datetime, excluded: np.ndarray | None = None
) -> UserRecommendations | None:
    """The fallback item: the catalog games with the most recent positive reviews (ties: lowest
    appid), scored by their share of the top game's count, never an `excluded` row (bool per
    catalog row). None without recent reviews."""
    games = data.games
    game_idx = np.flatnonzero(games.catalog.row_of >= 0)
    rows = games.catalog.row_of[game_idx]
    if excluded is not None:
        allowed = ~excluded[rows]
        game_idx, rows = game_idx[allowed], rows[allowed]
    counts = np.zeros(len(game_idx), dtype=np.int64)
    inside = game_idx < len(data.popular_counts)
    counts[inside] = data.popular_counts[game_idx[inside]]
    order = np.lexsort((games.game_id[rows], -counts))
    top = order[counts[order] > 0][:k]
    if not len(top):
        log.warning("no recent positive reviews: no popularity fallback item")
        return None
    best = float(counts[top[0]])
    return UserRecommendations(
        user_id=POPULAR_USER_ID,
        recommendations=[
            Recommendation(
                game_id=int(games.game_id[rows[i]]),
                name=str(games.name[rows[i]]),
                score=round(float(counts[i]) / best, 4),
            )
            for i in top
        ],
        model_id=POPULARITY_MODEL_ID,
        generated_at=now,
        reranked=False,
    )


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


def rerank_request(
    data: InferenceData, candidates: Candidates, user: int, excluded: np.ndarray | None = None
) -> RerankRequest:
    """The user's last `LIKED_GAMES` liked games (the start of what the user tower saw) + their
    candidates. Liked `excluded` (adult) games stay out of the prompt, so explanations never
    name them."""
    games = data.games
    liked = data.users.history[user][:LIKED_GAMES]
    liked_rows = games.catalog.rows(liked[liked != PADDING_ID])
    rows, _ = candidates.of(user)
    return RerankRequest(
        liked=[
            games.info(int(row))
            for row in liked_rows
            if row >= 0 and (excluded is None or not excluded[row])
        ],
        candidates=[games.info(int(row)) for row in rows],
        seed=int(data.users.user_id[user]),
    )


def liked_games(
    data: InferenceData, user: int, excluded: np.ndarray | None = None
) -> list[LikedGame]:
    """The user's last `LIKED_GAMES` liked games (the same as the rerank prompt), most recent
    first: catalog games only, never an `excluded` (adult / unreleased) one."""
    games = data.games
    history = data.users.history[user][:LIKED_GAMES]
    rows = games.catalog.rows(history[history != PADDING_ID])
    return [
        LikedGame(game_id=int(games.game_id[row]), name=str(games.name[row]))
        for row in rows
        if row >= 0 and (excluded is None or not excluded[row])
    ]


def user_recommendations(
    data: InferenceData,
    candidates: Candidates,
    reranked: dict[int, Reranked],
    *,
    model_id: str,
    rerank_model: str,
    now: datetime,
    excluded: np.ndarray | None = None,
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
            liked=liked_games(data, user, excluded),
        )
