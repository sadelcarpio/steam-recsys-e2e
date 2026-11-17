"""`python -m steam_training pair`: promote a (retriever, ranker) pair, or roll back (spec 9).

The serving pair is `evaluation/champion/pair.json` (before the first pair promotion: the
champion retriever without a ranker). A challenge names a retriever and / or a ranker:

- a new retriever R: the pairs (R, its ranker K) when `RANKER_ID` is given, (R, the serving
  ranker) when there is one, and (R, no ranker) are all evaluated;
- a new ranker only: (serving retriever, K).

Every pair is scored on the same query groups, at or after the newest of: each compared
retriever's split end (its cutoff without one) and each compared ranker's T2. So no compared
model trained, selected an epoch or stopped early on them. Gates:

1. a new retriever: retrieval recall@`PAIR_RETRIEVAL_K` beats the serving retriever's by
   `RETRIEVER_MIN_IMPROVEMENT`, and recall@`PAIR_CANDIDATES` (the ranker's ceiling) does not drop;
2. the final list: the best challenger pair's recall@`PAIR_TOP_K` beats the serving pair's by
   `MIN_IMPROVEMENT`, and its NDCG@`PAIR_TOP_K` does not drop.

A ranker must have passed its own gate (`ranker.json`) and use the current
`FEATURE_SCHEMA_VERSION`. Promotion copies the changed models into `models/champion/` /
`rankers/champion/` (none: the champion ranker is deleted), then appends history.jsonl and
writes pair.json last. `ROLLBACK=true` restores the pair that served before the current one was
promoted (each rollback steps one promotion back); run inference afterwards, as DynamoDB keeps
the current lists until the next run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from steam_training.artifacts import ArtifactStore
from steam_training.config import CHAMPION, PairSettings
from steam_training.contracts import (
    ModelMetadata,
    PairEvaluation,
    PairRecord,
    PairReport,
    RankerMetadata,
)
from steam_training.data import TableSource
from steam_training.evaluation import CatalogIndex, model_scores, top_candidates
from steam_training.ranker.data import epoch_seconds, load_ranker_data
from steam_training.ranker.features import FEATURE_SCHEMA_VERSION, SECONDS_PER_DAY
from steam_training.ranker.lists import candidate_lists, evaluate_lists, snapshots
from steam_training.ranker.model import (
    FitResult,
    build_groups,
    ranking_metrics,
    retrieval_score,
    target_positions,
)
from steam_training.train import resolve_device

log = logging.getLogger(__name__)

REPORT_KS = (10,)


@dataclass(frozen=True)
class Pair:
    retriever_id: str
    ranker_id: str | None

    def __str__(self) -> str:
        return f"{self.retriever_id} + {self.ranker_id or 'no ranker'}"


def run_pair(
    settings: PairSettings, source: TableSource, store: ArtifactStore
) -> PairReport | PairRecord:
    if settings.rollback:
        return rollback(store)
    serving = serving_pair(store)
    champion = Pair(serving.retriever_id, serving.ranker_id)
    retriever_id = settings.retriever_id or champion.retriever_id
    retriever_changed = retriever_id != champion.retriever_id
    rankers: list[str | None] = []
    if settings.ranker_id:
        rankers.append(settings.ranker_id)
    if retriever_changed:
        if champion.ranker_id and champion.ranker_id not in rankers:
            rankers.append(champion.ranker_id)
        rankers.append(None)
    challengers = [Pair(retriever_id, r) for r in dict.fromkeys(rankers)]
    if not challengers or challengers == [champion]:
        raise ValueError(f"nothing to challenge: {champion} is already serving")

    retrievers = {rid: store.load_model(rid) for rid in {champion.retriever_id, retriever_id}}
    rankers_ = {}
    for rid in {p.ranker_id for p in [champion, *challengers] if p.ranker_id}:
        rankers_[rid] = load_ranker(store, rid)
    if settings.ranker_id and not settings.force_promotion:
        report = store.read_ranker_report(settings.ranker_id)
        if report is None or not report.gate_passed:
            reason = report.reason if report else "no ranker.json"
            raise ValueError(f"ranker {settings.ranker_id} did not pass its gate: {reason}")

    start = window_start([m for _, m in retrievers.values()], [m for _, m in rankers_.values()])
    history_length = max(
        settings.pair_history_length, *(m.config.history_length for _, m in retrievers.values())
    )
    log.info("pairs: champion %s vs %s, groups from %s", champion, challengers, start)
    data = load_ranker_data(
        source,
        cutoff=start,
        history_length=history_length,
        max_rows=settings.pair_max_rows,
        max_groups_per_user=settings.pair_max_groups_per_user,
        seed=settings.seed,
        popularity_source=settings.pair_popularity_source,
        t2=start,
        list_users=settings.pair_list_users,
    )
    rows = data.rows
    if len(rows) > settings.pair_max_groups:
        rng = np.random.default_rng(settings.seed)
        rows = rows.take(np.sort(rng.choice(len(rows), settings.pair_max_groups, replace=False)))
    if not len(rows):
        raise ValueError(f"no query groups after {start} to compare the pairs on")

    device = resolve_device(settings.device)
    index = CatalogIndex.of(data.catalog, device)
    k, top_k = settings.pair_candidates, settings.pair_top_k
    ks = sorted({*REPORT_KS, top_k, settings.pair_retrieval_k, k})
    horizon = settings.pair_list_horizon_days * SECONDS_PER_DAY
    times = snapshots(
        epoch_seconds(start), int(rows.seconds[-1]), horizon, settings.pair_list_max_snapshots
    )
    scored: dict[str, tuple] = {}
    for rid, (model, _) in retrievers.items():
        score_fn = model_scores(model, data.catalog, settings.eval_batch_size, device)
        candidates, scores = top_candidates(
            score_fn, rows.history, index, k, settings.eval_batch_size, exclude=rows.reviewed
        )
        groups = build_groups(
            rows, candidates, scores, data.games, data.popularity, settings.pair_history_length
        )
        scored[rid] = (score_fn, groups)

    def evaluate(pair: Pair) -> PairEvaluation:
        score_fn, groups = scored[pair.retriever_id]
        fit = rankers_[pair.ranker_id][0] if pair.ranker_id else None
        retrieval = target_positions(retrieval_score(groups), groups)
        final = target_positions(fit.predict(groups), groups) if fit else retrieval
        user_lists = None
        if data.list_reviews is not None and len(data.list_reviews) and times:
            rank = candidate_lists(
                score_fn, index, k, settings.eval_batch_size, settings.pair_history_length,
                data.games, data.popularity, fit.predict_features if fit else None,
            )  # fmt: skip
            length = rows.history.shape[1]
            _, user_lists = evaluate_lists(data.list_reviews, times, horizon, length, ks, rank)
        evaluation = PairEvaluation(
            retriever_id=pair.retriever_id,
            ranker_id=pair.ranker_id,
            final=ranking_metrics(final, ks),
            retrieval=ranking_metrics(retrieval, ks),
            user_lists=user_lists,
        )
        log.info(
            "pair %s: recall@%d %.4f ndcg@%d %.4f (retrieval recall@%d %.4f, @%d %.4f)",
            pair, top_k, evaluation.final.recall[top_k], top_k, evaluation.final.ndcg[top_k],
            settings.pair_retrieval_k, evaluation.retrieval.recall[settings.pair_retrieval_k],
            k, evaluation.retrieval.recall[k],
        )  # fmt: skip
        return evaluation

    champion_eval = evaluate(champion)
    challenger_evals = [evaluate(p) for p in challengers]
    chosen, reason = decide_pair(champion_eval, challenger_evals, retriever_changed, settings)
    promoted = chosen is not None
    if not promoted and settings.force_promotion:
        chosen = max(challenger_evals, key=lambda e: e.final.recall[top_k])
        promoted, reason = True, f"forced (FORCE_PROMOTION): {reason}"
    report = PairReport(
        evaluated_at=datetime.now(UTC),
        window_start=start,
        top_k=top_k,
        candidates=k,
        champion=champion_eval,
        challengers=challenger_evals,
        chosen=chosen,
        promoted=promoted,
        reason=reason,
    )
    store.write_pair_report(report)
    log.info("pair promotion: %s (%s)", "PROMOTED" if promoted else "kept the pair", reason)
    if promoted and chosen is not None:
        if store.read_pair() is None:  # record what served before (the rollback target)
            store.write_pair(serving)
        promote_pair(
            store,
            Pair(chosen.retriever_id, chosen.ranker_id),
            current=champion,
            reason=reason,
            window_start=start,
            final=chosen.final,
        )
    return report


def decide_pair(
    champion: PairEvaluation,
    challengers: list[PairEvaluation],
    retriever_changed: bool,
    settings: PairSettings,
) -> tuple[PairEvaluation | None, str]:
    """The challenger pair to promote (None: keep the serving pair) and why."""
    k, rk, top = settings.pair_candidates, settings.pair_retrieval_k, settings.pair_top_k
    if retriever_changed:
        new, old = challengers[0].retrieval, champion.retrieval
        if not new.recall[rk] > old.recall[rk] + settings.retriever_min_improvement:
            return None, (
                f"retriever {challengers[0].retriever_id}: recall@{rk} {new.recall[rk]:.4f} does "
                f"not beat {old.recall[rk]:.4f} (+{settings.retriever_min_improvement})"
            )
        if new.recall[k] < old.recall[k]:
            return None, (
                f"retriever {challengers[0].retriever_id}: recall@{k} (the ranker's ceiling) "
                f"drops {old.recall[k]:.4f} -> {new.recall[k]:.4f}"
            )
    best = max(challengers, key=lambda e: (e.final.recall[top], e.final.ndcg[top]))
    name = f"{best.retriever_id} + {best.ranker_id or 'no ranker'}"
    recall, base = best.final.recall[top], champion.final.recall[top]
    ndcg, base_ndcg = best.final.ndcg[top], champion.final.ndcg[top]
    if recall > base + settings.min_improvement and ndcg >= base_ndcg:
        return best, f"{name}: recall@{top} {recall:.4f} > {base:.4f}, ndcg@{top} {ndcg:.4f}"
    return None, (
        f"best pair {name}: recall@{top} {recall:.4f} vs {base:.4f} "
        f"(+{settings.min_improvement}), ndcg@{top} {ndcg:.4f} vs {base_ndcg:.4f}"
    )


def serving_pair(store: ArtifactStore) -> PairRecord:
    """pair.json, or (before the first pair promotion) the champion retriever alone."""
    record = store.read_pair()
    if record is not None:
        return record
    metadata = store.load_metadata(CHAMPION)
    if metadata is None:
        raise FileNotFoundError("no champion retriever (models/champion/) to compare with")
    return PairRecord(
        retriever_id=metadata.model_id,
        ranker_id=None,
        promoted_at=datetime.now(UTC),
        reason="champion retriever before the first pair promotion",
    )


def window_start(retrievers: list[ModelMetadata], rankers: list[RankerMetadata]) -> datetime:
    """Groups from here on were never trained on or used for selection by any of them."""
    ends = [m.split.end or m.split.cutoff for m in retrievers]
    ends += [m.window.t2 for m in rankers]
    return max(ends)


def load_ranker(store: ArtifactStore, ranker_id: str) -> tuple[FitResult, RankerMetadata]:
    import lightgbm as lgb

    text, metadata = store.load_ranker(ranker_id)
    if metadata.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"ranker {ranker_id} has feature schema {metadata.feature_schema_version}, the code "
            f"computes {FEATURE_SCHEMA_VERSION}: retrain it"
        )
    booster = lgb.Booster(model_str=text)
    return FitResult(booster, list(metadata.feature_names), metadata.best_iteration), metadata


def promote_pair(
    store: ArtifactStore,
    pair: Pair,
    *,
    current: Pair,
    reason: str,
    window_start: datetime | None = None,
    final=None,
) -> PairRecord:
    if pair.retriever_id != current.retriever_id:
        store.copy_model_to_champion(pair.retriever_id)
    if pair.ranker_id != current.ranker_id:
        store.copy_ranker_to_champion(pair.ranker_id)
    record = PairRecord(
        retriever_id=pair.retriever_id,
        ranker_id=pair.ranker_id,
        promoted_at=datetime.now(UTC),
        reason=reason,
        window_start=window_start,
        final=final,
    )
    store.write_pair(record)
    return record


def rollback(store: ArtifactStore) -> PairRecord:
    """Back to the pair that served before the current one was promoted."""
    history = store.read_history()
    current = store.read_pair()
    if current is None or len(history) < 2:
        raise ValueError("no earlier pair to roll back to (history.jsonl has < 2 entries)")
    now = Pair(current.retriever_id, current.ranker_id)
    # the record that made the current pair serve: its last promotion (not a rollback to it)
    promoted_at = max(
        (
            i
            for i, r in enumerate(history)
            if Pair(r.retriever_id, r.ranker_id) == now and not r.reason.startswith("rollback")
        ),
        default=len(history) - 1,
    )
    if promoted_at == 0:
        raise ValueError(f"{now} is the first recorded pair: nothing to roll back to")
    target = history[promoted_at - 1]
    log.warning("rollback: %s -> %s + %s", now, target.retriever_id, target.ranker_id)
    return promote_pair(
        store,
        Pair(target.retriever_id, target.ranker_id),
        current=now,
        reason=f"rollback from {now} to the pair promoted at {target.promoted_at:%Y-%m-%d %H:%M}",
        window_start=target.window_start,
        final=target.final,
    )
