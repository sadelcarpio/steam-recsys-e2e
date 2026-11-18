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
    Concentration,
    ModelMetadata,
    PairEvaluation,
    PairRecord,
    PairReport,
    RankerMetadata,
)
from steam_training.data import Catalog, TableSource, current_catalog
from steam_training.evaluation import (
    CatalogIndex,
    list_concentration,
    model_scores,
    top_candidates,
    weekly_scores,
    weekly_top_candidates,
)
from steam_training.languages import load_review_languages
from steam_training.ranker.data import RankerRows, epoch_seconds, game_details, load_ranker_data
from steam_training.ranker.features import (
    FEATURE_SCHEMA_VERSION,
    SECONDS_PER_DAY,
    GameTable,
    compute_features,
)
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
    if settings.apply_report:
        return apply_report(store, settings.apply_report)
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
        # A6 retrievers score against the catalog as of each row's week; others ignore recency
        with_recency=any(m.config.item_recency for m, _ in retrievers.values()),
        # A7 retrievers read each group's previous review language; others ignore it
        languages=load_review_languages(source)
        if any(m.config.user_language for m, _ in retrievers.values())
        else None,
    )
    rows = data.rows
    if len(rows) > settings.pair_max_groups:
        rng = np.random.default_rng(settings.seed)
        rows = rows.take(np.sort(rng.choice(len(rows), settings.pair_max_groups, replace=False)))
    if not len(rows):
        raise ValueError(f"no query groups after {start} to compare the pairs on")

    device = resolve_device(settings.device)
    # loaded on the CPU; scored on `device` (a GPU when DEVICE=cuda)
    retrievers = {rid: (model.to(device).eval(), meta) for rid, (model, meta) in retrievers.items()}
    index = CatalogIndex.of(data.catalog, device)
    k, top_k = settings.pair_candidates, settings.pair_top_k
    ks = sorted({*REPORT_KS, top_k, settings.pair_retrieval_k, k})
    horizon = settings.pair_list_horizon_days * SECONDS_PER_DAY
    times = snapshots(
        epoch_seconds(start), int(rows.seconds[-1]), horizon, settings.pair_list_max_snapshots
    )
    scored: dict[str, tuple] = {}
    for rid, (model, _) in retrievers.items():
        scores_at = weekly_scores(model, data.catalog, settings.eval_batch_size, device)
        candidates, scores = weekly_top_candidates(
            scores_at,
            rows.history,
            rows.seconds,
            index,
            k,
            settings.eval_batch_size,
            exclude=rows.reviewed,
            language=rows.language,
        )
        groups = build_groups(
            rows, candidates, scores, data.games, data.popularity, settings.pair_history_length
        )
        scored[rid] = (scores_at, groups)

    # exposure over the current catalog for the most recent groups' users (the gate)
    current = current_catalog(
        source,
        data.vocab.games,
        with_tags=data.vocab.tags is not None,
        with_recency=any(m.config.item_recency for m, _ in retrievers.values()),
        snapshots=dict(data.snapshots),
    )
    recent = rows.take(np.arange(max(len(rows) - settings.pair_concentration_rows, 0), len(rows)))
    exposure, final_exposure = current_exposure(
        settings, source, data, current, recent, retrievers, rankers_, [champion, *challengers],
        device,
    )  # fmt: skip

    def evaluate(pair: Pair) -> PairEvaluation:
        scores_at, groups = scored[pair.retriever_id]
        fit = rankers_[pair.ranker_id][0] if pair.ranker_id else None
        retrieval = target_positions(retrieval_score(groups), groups)
        final = target_positions(fit.predict(groups), groups) if fit else retrieval
        user_lists = None
        if data.list_reviews is not None and len(data.list_reviews) and times:
            rank = candidate_lists(
                scores_at, index, k, settings.eval_batch_size, settings.pair_history_length,
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
            concentration=exposure[pair.retriever_id],
            final_concentration=final_exposure[pair],
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
    if chosen is None and settings.force_promotion:
        chosen = max(challenger_evals, key=lambda e: e.final.recall[top_k])
        reason = f"forced (FORCE_PROMOTION): {reason}"
    promoted = chosen is not None and not settings.pair_dry_run
    if settings.pair_dry_run:
        reason = f"dry run ({'would promote' if chosen else 'nothing to promote'}): {reason}"
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
        dry_run=settings.pair_dry_run,
    )
    store.write_pair_report(report, settings.pair_report_key)
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


def apply_report(store: ArtifactStore, key: str) -> PairRecord:
    """Promote the pair a dry-run report chose, if the serving pair is still the one it was
    compared with."""
    report = store.read_pair_report(key)
    if report is None:
        raise FileNotFoundError(f"no pair report at {key}")
    if not report.dry_run or report.chosen is None:
        raise ValueError(f"{key} is not a dry-run report with a chosen pair: {report.reason}")
    serving = serving_pair(store)
    champion = report.champion
    if champion is None or (serving.retriever_id, serving.ranker_id) != (
        champion.retriever_id,
        champion.ranker_id,
    ):
        raise ValueError(
            f"the serving pair is now {serving.retriever_id} + {serving.ranker_id}, the report "
            "compared another one: evaluate again"
        )
    if store.read_pair() is None:
        store.write_pair(serving)
    chosen = report.chosen
    return promote_pair(
        store,
        Pair(chosen.retriever_id, chosen.ranker_id),
        current=Pair(serving.retriever_id, serving.ranker_id),
        reason=f"approved ({key}): {report.reason.removeprefix('dry run (would promote): ')}",
        window_start=report.window_start,
        final=chosen.final,
    )


def current_exposure(
    settings: PairSettings,
    source: TableSource,
    data,
    current: Catalog,
    recent: RankerRows,
    retrievers: dict[str, tuple],
    rankers: dict[str, tuple],
    pairs: list[Pair],
    device,
) -> tuple[dict[str, Concentration], dict[Pair, Concentration]]:
    """Each retriever's top-`PAIR_CONCENTRATION_K` lists over the current catalog for the
    recent groups' users, and each pair's final lists (its ranker re-orders the retriever's
    top `PAIR_CANDIDATES` as of the newest group, on features of the current catalog)."""
    k, top = settings.pair_candidates, settings.pair_concentration_k
    snapshots = dict(data.snapshots)
    for table in ("lkp_games", "game_details"):
        snapshots.setdefault(table, source.snapshot_id(table))
    price, release_day = game_details(source, snapshots, data.vocab.games)
    games = GameTable.from_catalog(
        current, data.vocab.games, data.vocab.tags or 0, data.vocab.genres,
        data.vocab.categories, price=price, release_day=release_day,
    )  # fmt: skip
    index = CatalogIndex.of(current, device)
    now = np.full(len(recent), int(recent.seconds.max()) if len(recent) else 0, np.int64)
    history = recent.history[:, : settings.pair_history_length]
    exposure: dict[str, Concentration] = {}
    final: dict[Pair, Concentration] = {}
    for rid, (model, _) in retrievers.items():
        seen = model.item_tower.game_ids.seen_games.detach().cpu().numpy()
        candidates, scores = top_candidates(
            model_scores(model, current, settings.eval_batch_size, device),
            recent.history,
            index,
            k,
            settings.eval_batch_size,
            exclude=recent.reviewed,
            language=recent.language,
        )
        retrieved = np.where(np.isfinite(scores), candidates, -1)
        exposure[rid] = list_concentration(retrieved[:, :top], seen)
        for pair in (p for p in pairs if p.retriever_id == rid):
            if pair.ranker_id is None:
                final[pair] = exposure[rid]
                continue
            fit = rankers[pair.ranker_id][0]
            ranked = np.empty(scores.shape, np.float64)
            for start in range(0, len(recent), 2048):
                part = slice(start, start + 2048)
                x = compute_features(
                    history[part], candidates[part], scores[part], now[part], games,
                    data.popularity,
                )  # fmt: skip
                ranked[part] = fit.predict_features(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
            ranked[retrieved < 0] = -np.inf
            order = np.argsort(-ranked, axis=1, kind="stable")[:, :top]
            final[pair] = list_concentration(np.take_along_axis(retrieved, order, axis=1), seen)
        for pair in (p for p in pairs if p.retriever_id == rid):
            log.info(
                "%s over the current catalog: top game (game_idx %d) in %.1f%% of top-%d lists "
                "(retrieval order: %.1f%%), unseen games %.1f%% of slots",
                pair, final[pair].top_game_idx, 100 * final[pair].reach, top,
                100 * exposure[rid].reach, 100 * final[pair].unseen_share,
            )  # fmt: skip
    return exposure, final


def decide_pair(
    champion: PairEvaluation,
    challengers: list[PairEvaluation],
    retriever_changed: bool,
    settings: PairSettings,
) -> tuple[PairEvaluation | None, str]:
    """The challenger pair to promote (None: keep the serving pair) and why."""
    k, rk, top = settings.pair_candidates, settings.pair_retrieval_k, settings.pair_top_k
    if retriever_changed:
        exposure = challengers[0].concentration
        if exposure is not None and exposure.reach > settings.pair_max_game_reach:
            return None, (
                f"retriever {challengers[0].retriever_id}: its most common game (game_idx "
                f"{exposure.top_game_idx}) is in {exposure.reach:.1%} of top-{exposure.k} lists "
                f"over the current catalog (> PAIR_MAX_GAME_REACH {settings.pair_max_game_reach})"
            )
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
    flooding = [
        e for e in challengers
        if e.final_concentration is not None
        and e.final_concentration.reach > settings.pair_max_game_reach
    ]  # fmt: skip
    challengers = [e for e in challengers if e not in flooding]
    if not challengers:
        worst = max(flooding, key=lambda e: e.final_concentration.reach)
        c = worst.final_concentration
        return None, (
            f"every challenger floods one game: {worst.retriever_id} + "
            f"{worst.ranker_id or 'no ranker'} sends game_idx {c.top_game_idx} to {c.reach:.1%} "
            f"of top-{c.k} lists over the current catalog (> PAIR_MAX_GAME_REACH "
            f"{settings.pair_max_game_reach})"
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
