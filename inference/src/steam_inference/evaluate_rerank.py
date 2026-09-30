"""Rerank evaluation (spec 7): does the LLM order beat the retrieval order?

    AWS_PROFILE=<admin> USE_SSM=true uv run python -m steam_inference.evaluate_rerank [--users 500]

For a sample of users active after the model's training cutoff (at least `RERANK_MIN_REVIEWS`
reviews before it and a positive review at or after it), the history is rebuilt as of the cutoff
(latest `user_features` row and reviewed games before it), the model retrieves `TOP_K`
candidates, and stage 1 of the rerank orders them. Hit rate@5 / @10 against the games each user
reviewed positively from the cutoff on is reported for the retrieval order, the LLM order and the
blends of both. Game features are the current ones (a small leak, the same for every order).

Writes `s3://<bucket>/evaluation/rerank/<model_id>/<timestamp>.json` and prints a table. Cost
bound: one ranking call per sampled user (~$1 per 500 users with Nova 2 Lite).
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
from steam_training.artifacts import ArtifactStore
from steam_training.data import TableSource

from steam_inference.adult import adult_mask
from steam_inference.config import InferenceSettings, configure_logging
from steam_inference.contracts import RerankEvaluation
from steam_inference.features import (
    REVIEW_COLUMNS,
    TABLES,
    InferenceData,
    Reviews,
    Users,
    collect_reviews,
    latest_users,
    load_games,
    member,
    review_activity,
    review_rows,
)
from steam_inference.pipeline import rerank_request
from steam_inference.rerank import BedrockRerankLlm, RerankLlm, blend, llm_ranking
from steam_inference.retrieval import retrieve

log = logging.getLogger(__name__)

WEIGHTS = (0.25, 0.5, 0.75)
CUTS = (5, 10)


def hit_rates(orders: list[list[int]], truth: list[set[int]], cuts=CUTS) -> dict[str, float]:
    """Share of users whose first k entries contain a truth item, per k."""
    return {
        f"hit@{k}": float(
            np.mean([bool(set(order[:k]) & t) for order, t in zip(orders, truth, strict=True)])
        )
        if orders
        else 0.0
        for k in cuts
    }


def compare_orders(
    llm_orders: list[list[int]], truth: list[set[int]], weights=WEIGHTS
) -> dict[str, dict[str, float]]:
    """Hit rates of the retrieval order (candidate positions 0..K-1), the LLM order and blends.
    `truth`: per user, the candidate positions reviewed positively after the cutoff."""
    orders = {
        "retrieval": [list(range(len(o))) for o in llm_orders],
        "llm": llm_orders,
        **{f"blend_{w}": [blend(o, w) for o in llm_orders] for w in weights},
    }
    return {name: hit_rates(ranked, truth) for name, ranked in orders.items()}


def to_micros(moment: datetime) -> int:
    """Microseconds since the epoch (naive datetimes are UTC, as in the marts)."""
    aware = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    return (aware - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def sample_cohort(
    source: TableSource, snapshot_id: int | None, cutoff: int, *, min_reviews: int, size: int, seed
) -> np.ndarray:
    """Sorted ids of up to `size` random users with >= `min_reviews` reviews (one positive)
    before `cutoff` and a positive review at or after it."""
    before = review_activity(source, snapshot_id, until=cutoff - 1)
    active = before.user_id[(before.reviews >= min_reviews) & (before.positives > 0)]
    later: list[np.ndarray] = []
    for batch in source.batches("interactions", REVIEW_COLUMNS, snapshot_id):
        user, _, ts, positive = review_rows(batch)
        later.append(np.unique(user[positive & (ts >= cutoff)]))
    after = np.unique(np.concatenate(later)) if later else np.zeros(0, dtype=np.int64)
    eligible = active[member(active, after)]
    if len(eligible) > size:
        eligible = np.sort(np.random.default_rng(seed).choice(eligible, size, replace=False))
    return eligible


def load_cohort(
    source: TableSource, snapshots: dict, cohort: np.ndarray, cutoff: int
) -> tuple[Users, Reviews, dict[int, set[int]]]:
    """The cohort's users and reviewed games as of the cutoff, and their later positives."""
    review_users, reviews, _ = collect_reviews(
        source, snapshots["interactions"], cohort, popular_since=0, until=cutoff - 1
    )
    after: dict[int, set[int]] = {}
    for batch in source.batches("interactions", REVIEW_COLUMNS, snapshots["interactions"]):
        user, game, ts, positive = review_rows(batch)
        keep = positive & (ts >= cutoff) & member(user, cohort)
        for u, g in zip(user[keep].tolist(), game[keep].tolist(), strict=True):
            after.setdefault(u, set()).add(g)
    users = latest_users(
        source.batches(
            "user_features",
            ["user_id", "timestamp", "games_reviewed_positive"],
            snapshots["user_features"],
        ),
        keep=cohort,
        until=cutoff - 1,
    )
    start = np.searchsorted(review_users, users.user_id, side="left")
    count = np.searchsorted(review_users, users.user_id, side="right") - start
    return Users(users.user_id, users.history, start, count), reviews, after


def evaluate_rerank(
    settings: InferenceSettings,
    source: TableSource,
    store: ArtifactStore,
    llm: RerankLlm,
    *,
    users: int = 500,
    seed: int = 0,
    now: datetime | None = None,
) -> RerankEvaluation:
    now = now or datetime.now(UTC)
    model, metadata = store.load_model(settings.model_id)
    cutoff = to_micros(metadata.split.cutoff)
    snapshots = {table: source.snapshot_id(table) for table in TABLES}
    games = load_games(source, snapshots, settings.rerank_description_chars)
    cohort = sample_cohort(
        source,
        snapshots["interactions"],
        cutoff,
        min_reviews=settings.rerank_min_reviews,
        size=users,
        seed=seed,
    )
    cohort_users, reviews, after = load_cohort(source, snapshots, cohort, cutoff)
    log.info("evaluating %d users active after %s", len(cohort_users), metadata.split.cutoff)
    excluded = adult_mask(games) if settings.exclude_adult else np.zeros(len(games), bool)
    if games.coming_soon is not None:
        excluded |= games.coming_soon
    candidates = retrieve(
        model,
        games,
        cohort_users,
        reviews,
        k=settings.top_k,
        user_batch_size=settings.user_batch_size,
        item_batch_size=settings.item_batch_size,
        excluded=excluded,
    )
    data = InferenceData(games, cohort_users, reviews, np.zeros(0, np.int64), snapshots)
    ranked = [u for u in range(len(cohort_users)) if len(candidates.of(u)[0]) >= 2]

    def rank(user: int) -> list[int] | None:
        request = rerank_request(data, candidates, user, excluded)
        return llm_ranking(llm, request, settings.rerank_shuffle)[0]

    with ThreadPoolExecutor(max_workers=settings.rerank_concurrency) as pool:
        answers = list(pool.map(rank, ranked))
    llm_orders, truth, fallbacks = [], [], 0
    for user, order in zip(ranked, answers, strict=True):
        if order is None:
            fallbacks += 1
            continue
        rows, _ = candidates.of(user)
        later = after.get(int(cohort_users.user_id[user]), set())
        game_idx = games.catalog.items.game_idx[rows]
        truth.append({p for p, g in enumerate(game_idx.tolist()) if g in later})
        llm_orders.append(order)
    usage = getattr(llm, "usage", {}).get("rank")
    report = RerankEvaluation(
        model_id=metadata.model_id,
        rerank_model=settings.bedrock_model_id,
        cutoff=metadata.split.cutoff,
        generated_at=now,
        sampled_users=len(cohort_users),
        evaluated_users=len(llm_orders),
        rank_fallbacks=fallbacks,
        top_k=settings.top_k,
        hit_rates=compare_orders(llm_orders, truth),
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
    )
    key = f"evaluation/rerank/{metadata.model_id}/{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    store.s3.put_object(
        Bucket=store.bucket,
        Key=key,
        Body=report.model_dump_json(indent=2).encode(),
        ContentType="application/json",
    )
    log.info("rerank evaluation written to s3://%s/%s", store.bucket, key)
    return report


def format_table(report: RerankEvaluation) -> str:
    cuts = [f"hit@{k}" for k in CUTS]
    lines = [f"{'order':<14}" + "".join(f"{c:>9}" for c in cuts)]
    for name, rates in report.hit_rates.items():
        lines.append(f"{name:<14}" + "".join(f"{rates[c]:>9.3f}" for c in cuts))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", type=int, default=500, help="sampled users")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    settings = InferenceSettings()
    configure_logging(settings.log_level)
    from steam_training.data import IcebergSource

    llm = BedrockRerankLlm(
        settings.bedrock_model_id,
        max_tokens=settings.rerank_max_tokens,
        temperature=settings.rerank_temperature,
        region=settings.aws_region,
        concurrency=settings.rerank_concurrency,
    )
    report = evaluate_rerank(
        settings,
        IcebergSource(settings.glue_database, settings.aws_region),
        ArtifactStore(settings.model_artifacts_bucket),
        llm,
        users=args.users,
        seed=args.seed,
    )
    print(
        f"model {report.model_id}, {report.evaluated_users}/{report.sampled_users} users "
        f"({report.rank_fallbacks} ranking fallbacks)\n{format_table(report)}"
    )


if __name__ == "__main__":
    main()
