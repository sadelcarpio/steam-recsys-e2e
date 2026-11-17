"""`python -m steam_training rank`: train and report a ranker on a retriever's candidates.

1. Load the retriever (`RETRIEVER_ID`, default the champion) and its cutoff T1.
2. Query groups after T1 (`ranker.data`), sorted by time; T2 splits them by
   `RANKER_TRAIN_FRACTION` into train (before T2) and test (from T2).
3. The retriever's top `RANKER_CANDIDATES` per group (games reviewed before t excluded).
4. Features (`ranker.features`), LightGBM `lambdarank` on the training groups that contain
   their game; the last `RANKER_EARLY_STOPPING_FRACTION` of them (by time) stops it early.
5. Report on the test groups (all of them; a missed game is a miss): retrieval order vs ranker
   order, the candidates' recall (the ceiling), a split by history size, gain importances and,
   with `RANKER_ABLATION`, the change when each feature family is left out.
6. Gate: recall@top_k must beat retrieval order by `RANKER_MIN_GAIN`. The ranker is saved either
   way (experiments), and the report says whether it passed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from steam_training.artifacts import ArtifactStore
from steam_training.config import RankerSettings
from steam_training.contracts import (
    RankerComparison,
    RankerMetadata,
    RankerReport,
    RankerWindow,
    UserListComparison,
)
from steam_training.data import TableSource
from steam_training.evaluation import CatalogIndex, model_scores, top_candidates
from steam_training.ranker.data import epoch_seconds, from_epoch_seconds, load_ranker_data
from steam_training.ranker.features import (
    FAMILIES,
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    SECONDS_PER_DAY,
    compute_features,
)
from steam_training.ranker.lists import SnapshotLists, evaluate_lists
from steam_training.ranker.lists import snapshots as list_snapshots
from steam_training.ranker.model import (
    FitResult,
    Groups,
    build_groups,
    fit_lambdarank,
    ranking_metrics,
    retrieval_score,
    target_positions,
)
from steam_training.train import resolve_device

log = logging.getLogger(__name__)

REPORT_KS = (10,)  # reported next to RANKER_TOP_K


def run_ranking(
    settings: RankerSettings, source: TableSource, store: ArtifactStore
) -> RankerReport:
    retriever, retriever_meta = store.load_model(settings.retriever_id)
    retriever_id = retriever_meta.model_id
    cutoff = retriever_meta.split.cutoff
    history_length = max(retriever_meta.config.history_length, settings.ranker_history_length)
    log.info("ranker %s on retriever %s (cutoff %s)", settings.ranker_id, retriever_id, cutoff)

    data = load_ranker_data(
        source,
        cutoff=cutoff,
        history_length=history_length,
        max_rows=settings.ranker_max_rows,
        max_groups_per_user=settings.ranker_max_groups_per_user,
        seed=settings.seed,
        popularity_source=settings.ranker_popularity_source,
        t2=retriever_meta.split.end,
        train_fraction=settings.ranker_train_fraction,
        list_users=settings.ranker_list_users,
    )
    rows = data.rows
    cut, t2 = data.test_start, data.t2
    if cut < 1 or cut >= len(rows):
        raise ValueError(f"no ranker groups on both sides of T2 {t2} (cutoff {cutoff})")
    rng = np.random.default_rng(settings.seed)
    train_idx = _subsample(np.arange(cut), settings.ranker_max_train_groups, rng)
    test_idx = _subsample(np.arange(cut, len(rows)), settings.ranker_max_test_groups, rng)

    device = resolve_device(settings.device)
    score_fn = model_scores(retriever, data.catalog, settings.eval_batch_size, device)
    index = CatalogIndex.of(data.catalog, device)
    k = settings.ranker_candidates

    def groups_of(idx: np.ndarray) -> Groups:
        part = rows.take(idx)
        candidates, scores = top_candidates(
            score_fn, part.history, index, k, settings.eval_batch_size, exclude=part.reviewed
        )
        return build_groups(
            part, candidates, scores, data.games, data.popularity, settings.ranker_history_length
        )

    train_all = groups_of(train_idx)
    train = train_all.take(np.flatnonzero(train_all.has_target))
    test = groups_of(test_idx)
    if len(train) < settings.ranker_min_groups:
        raise ValueError(
            f"{len(train)} training groups with their game among the candidates "
            f"(< RANKER_MIN_GROUPS={settings.ranker_min_groups}): raise the retriever's "
            "VALIDATION_FRACTION or RANKER_MAX_ROWS"
        )
    early = int(round(len(train) * settings.ranker_early_stopping_fraction))
    window = RankerWindow(
        retriever_cutoff=cutoff,
        t2=t2,
        end=from_epoch_seconds(int(rows.seconds[-1])),
        train_groups=len(train) - early,
        early_stopping_groups=early,
        test_groups=len(test),
    )
    log.info(
        "groups: %d train (of %d; the others miss their game), %d test; T2 %s",
        len(train),
        len(train_all),
        len(test),
        t2,
    )

    top_k = settings.ranker_top_k
    ks = sorted({*REPORT_KS, top_k})
    fit = _fit(settings, train, FEATURES)
    ranker_position = target_positions(fit.predict(test), test)
    retrieval_position = target_positions(retrieval_score(test), test)
    comparison = RankerComparison(
        retrieval=ranking_metrics(retrieval_position, ks),
        ranker=ranking_metrics(ranker_position, ks),
    )
    found = test.has_target
    conditional = RankerComparison(
        retrieval=ranking_metrics(retrieval_position[found], ks),
        ranker=ranking_metrics(ranker_position[found], ks),
    )
    user_lists = None
    if data.list_reviews is not None and len(data.list_reviews):
        user_lists = _user_lists(settings, data, fit, score_fn, index, ks)
    by_history = {}
    full = test.history_size >= settings.ranker_history_length
    for name, mask in (("short", ~full), ("full", full)):
        by_history[name] = RankerComparison(
            retrieval=ranking_metrics(retrieval_position[mask], ks),
            ranker=ranking_metrics(ranker_position[mask], ks),
        )
    gains = fit.booster.feature_importance(importance_type="gain", iteration=fit.best_iteration)
    total = float(np.sum(gains)) or 1.0
    importances = {f: float(g) / total for f, g in zip(fit.features, gains, strict=True)}

    ablation: dict[str, float] = {}
    if settings.ranker_ablation:
        base = comparison.ranker.recall[top_k]
        for family, names in FAMILIES.items():
            kept = [f for f in FEATURES if f not in names]
            reduced = _fit(settings, train, kept)
            recall = ranking_metrics(target_positions(reduced.predict(test), test), [top_k])
            ablation[family] = recall.recall[top_k] - base
            log.info("without %s: recall@%d %+.4f", family, top_k, ablation[family])

    gain = comparison.ranker.recall[top_k] - comparison.retrieval.recall[top_k]
    passed = gain >= settings.ranker_min_gain
    reason = (
        f"recall@{top_k} {comparison.ranker.recall[top_k]:.4f} vs retrieval order "
        f"{comparison.retrieval.recall[top_k]:.4f} ({gain:+.4f}, needs "
        f"+{settings.ranker_min_gain})"
    )
    now = datetime.now(UTC)
    report = RankerReport(
        ranker_id=settings.ranker_id,
        retriever_id=retriever_id,
        evaluated_at=now,
        top_k=top_k,
        window=window,
        candidate_recall=float(test.has_target.mean()) if len(test) else 0.0,
        test=comparison,
        conditional=conditional,
        user_lists=user_lists,
        by_history=by_history,
        importances=dict(sorted(importances.items(), key=lambda kv: -kv[1])),
        ablation=ablation,
        gate_passed=passed,
        reason=reason,
    )
    metadata = RankerMetadata(
        ranker_id=settings.ranker_id,
        created_at=now,
        retriever_id=retriever_id,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(fit.features),
        candidates=k,
        top_k=top_k,
        history_length=settings.ranker_history_length,
        window=window,
        snapshots=data.snapshots,
        hyperparameters=_hyperparameters(settings),
        best_iteration=fit.best_iteration,
        test=comparison,
    )
    store.save_ranker(fit.booster.model_to_string(num_iteration=fit.best_iteration), metadata)
    store.write_ranker_report(report)
    log.info("ranker metrics %s", metrics_line(report))
    log.info("ranker gate: %s (%s)", "PASSED" if passed else "failed", reason)
    return report


def metrics_line(report: RankerReport) -> str:
    """One parseable line (CloudWatch metric definitions of the SageMaker job)."""
    k = report.top_k
    t = report.test
    return (
        f"retrieval_recall_at_k={t.retrieval.recall[k]:.6f} "
        f"ranker_recall_at_k={t.ranker.recall[k]:.6f} "
        f"retrieval_ndcg_at_k={t.retrieval.ndcg[k]:.6f} "
        f"ranker_ndcg_at_k={t.ranker.ndcg[k]:.6f} "
        f"candidate_recall={report.candidate_recall:.6f}"
        + (
            f" ranker_list_map_at_k={report.user_lists.ranker.map[k]:.6f}"
            f" retrieval_list_map_at_k={report.user_lists.retrieval.map[k]:.6f}"
            if report.user_lists is not None
            else ""
        )
    )


def _user_lists(settings, data, fit, score_fn, index, ks) -> UserListComparison:
    """Per-user lists at weekly snapshots of the test window (`ranker.lists`)."""
    horizon = settings.ranker_list_horizon_days * SECONDS_PER_DAY
    end = int(data.rows.seconds[-1])
    times = list_snapshots(epoch_seconds(data.t2), end, horizon, settings.ranker_list_max_snapshots)
    length = data.rows.history.shape[1]

    def rank(lists: SnapshotLists, t: int) -> tuple[np.ndarray, np.ndarray]:
        candidates, scores = top_candidates(
            score_fn, lists.history, index, settings.ranker_candidates,
            settings.eval_batch_size, exclude=lists.reviewed,
        )  # fmt: skip
        history = lists.history[:, : settings.ranker_history_length]
        seconds = np.full(len(lists), t, dtype=np.int64)
        ranked = np.empty_like(scores)
        for start in range(0, len(lists), 256):
            part = slice(start, start + 256)
            x = compute_features(
                history[part], candidates[part], scores[part], seconds[part],
                data.games, data.popularity,
            )  # fmt: skip
            ranked[part] = fit.predict_features(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
        missing = ~np.isfinite(scores)
        by_retrieval = np.where(missing, -1, candidates)
        ranked[missing] = -np.inf
        order = np.argsort(-ranked, axis=1, kind="stable")
        by_ranker = np.where(missing, -1, candidates)[np.arange(len(lists))[:, None], order]
        return by_retrieval, by_ranker

    retrieval, ranker = evaluate_lists(data.list_reviews, times, horizon, length, ks, rank)
    log.info(
        "per-user lists: %d (user, snapshot) lists over %d snapshots", ranker.lists, len(times)
    )
    return UserListComparison(
        snapshots=[from_epoch_seconds(t) for t in times],
        horizon_days=settings.ranker_list_horizon_days,
        retrieval=retrieval,
        ranker=ranker,
    )


def _fit(settings: RankerSettings, train: Groups, features: list[str]) -> FitResult:
    return fit_lambdarank(
        train,
        features,
        early_stopping_fraction=settings.ranker_early_stopping_fraction,
        top_k=settings.ranker_top_k,
        num_leaves=settings.ranker_num_leaves,
        learning_rate=settings.ranker_learning_rate,
        max_rounds=settings.ranker_max_rounds,
        early_stopping_rounds=settings.ranker_early_stopping_rounds,
        min_data_in_leaf=settings.ranker_min_data_in_leaf,
        feature_fraction=settings.ranker_feature_fraction,
        num_threads=settings.num_threads,
        seed=settings.seed,
    )


def _subsample(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """At most `n` of `idx`, keeping their (time) order."""
    if len(idx) <= n:
        return idx
    return np.sort(rng.choice(idx, n, replace=False))


def _hyperparameters(settings: RankerSettings) -> dict[str, object]:
    return settings.model_dump(
        mode="json",
        exclude={"model_artifacts_bucket", "ranker_id", "glue_database", "aws_region",
                 "data_cache_dir", "log_level"},
    )  # fmt: skip
