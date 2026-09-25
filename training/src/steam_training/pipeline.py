"""The two jobs: `train` (split -> loader -> training -> evaluation -> artifacts) and `promote`
(evaluate a trained model against the champion on common validation rows, swap if better)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import torch

from steam_training.artifacts import ArtifactStore
from steam_training.batching import build_examples
from steam_training.config import CHAMPION, TrainingSettings
from steam_training.contracts import (
    ARCHITECTURE_VERSION,
    EvaluationReport,
    ModelEvaluation,
    ModelMetadata,
    RecallMetrics,
)
from steam_training.data import TableSource, TrainingData, load_training_data
from steam_training.evaluation import evaluate, model_scores, popularity_scores
from steam_training.model import TwoTowerModel
from steam_training.train import fingerprint, resolve_device, train_model

log = logging.getLogger(__name__)

# Settings recorded in the metadata as hyperparameters (not infrastructure / identity).
_NOT_HYPERPARAMETERS = {"model_artifacts_bucket", "model_id", "glue_database", "aws_region"}


def _load(settings: TrainingSettings, source: TableSource, **kwargs: object) -> TrainingData:
    return load_training_data(
        source,
        validation_fraction=settings.validation_fraction,
        cold_row_fraction=settings.cold_row_fraction,
        eval_max_rows=settings.eval_max_rows,
        seed=settings.seed,
        **kwargs,
    )


def _evaluate_all(
    settings: TrainingSettings,
    data: TrainingData,
    models: dict[str, TwoTowerModel],
    device: torch.device,
) -> tuple[dict[str, RecallMetrics], RecallMetrics]:
    """Recall of each model and of the popularity baseline on the validation rows."""

    def run(score_fn) -> RecallMetrics:
        return evaluate(
            score_fn,
            data.validation,
            data.catalog,
            settings.recall_ks,
            settings.eval_batch_size,
            device,
        )

    results = {
        name: run(model_scores(model.to(device), data.catalog, settings.eval_batch_size, device))
        for name, model in models.items()
    }
    baseline = run(popularity_scores(data.train_positive_counts, data.catalog, device))
    return results, baseline


def run_training(
    settings: TrainingSettings, source: TableSource, store: ArtifactStore
) -> ModelMetadata:
    if settings.num_threads:
        torch.set_num_threads(settings.num_threads)
    device = resolve_device(settings.device)

    # 1. dataset + temporal split (streamed, filtered)
    data = _load(settings, source)
    assert data.train is not None

    # 2. training examples (the DataLoader / collate is built in train_model)
    examples = build_examples(
        data.train, data.catalog, data.negatives if settings.explicit_negatives else None
    )
    if len(examples) == 0:
        raise ValueError("no positive training rows")
    monitor_rows = (
        data.validation.sample(settings.epoch_eval_rows) if settings.epoch_eval_rows else None
    )

    # 3. training (checkpointed per epoch, resumed when the same run restarts)
    checkpoints = store.checkpoints(settings.model_id)
    model, losses = train_model(
        settings,
        examples,
        data.catalog,
        data.vocab,
        monitor_rows,
        device=device,
        checkpoints=checkpoints,
        run_fingerprint=fingerprint(settings, data.snapshots, data.split.cutoff),
    )

    # 4. evaluation
    results, baseline = _evaluate_all(settings, data, {"model": model}, device)
    log.info("validation %s", results["model"].model_dump_json())
    log.info("popularity baseline %s", baseline.model_dump_json())
    log.info(
        "metrics %s",
        metrics_line(settings.primary_k, {"final": results["model"], "popularity": baseline}),
    )

    metadata = ModelMetadata(
        model_id=settings.model_id,
        created_at=datetime.now(UTC),
        config=model.config,
        split=data.split,
        snapshots=data.snapshots,
        hyperparameters=settings.model_dump(exclude=_NOT_HYPERPARAMETERS),
        validation=results["model"],
        popularity_baseline=baseline,
        epoch_losses=losses,
    )
    # The checkpoint is kept: re-running this MODEL_ID with more EPOCHS extends the run.
    store.save_model(model, metadata)
    return metadata


def metrics_line(primary_k: int, metrics: dict[str, RecallMetrics]) -> str:
    """`<name>_warm_recall=<x> <name>_all_recall=<x> ...` at `primary_k`. A fixed format that the
    SageMaker metric definitions (`launch.METRIC_DEFINITIONS`) parse into CloudWatch series."""
    return " ".join(
        f"{name}_{segment}_recall={getattr(m, segment).recall[primary_k]:.6f}"
        for name, m in metrics.items()
        for segment in ("warm", "all")
    )


def _load_champion(
    store: ArtifactStore, candidate_id: str
) -> tuple[TwoTowerModel | None, ModelMetadata | None]:
    metadata = store.load_metadata(CHAMPION)
    if metadata is None or metadata.model_id == candidate_id:
        return None, metadata
    if metadata.config.architecture_version != ARCHITECTURE_VERSION:
        log.warning(
            "champion %s has architecture v%d (code is v%d): comparing with its stored metrics",
            metadata.model_id,
            metadata.config.architecture_version,
            ARCHITECTURE_VERSION,
        )
        return None, metadata
    model, metadata = store.load_model(CHAMPION)
    return model, metadata


def decide(
    primary_k: int,
    candidate: RecallMetrics,
    champion: RecallMetrics | None,
    baseline: RecallMetrics,
    min_improvement: float,
) -> tuple[bool, str]:
    score = candidate.warm.recall[primary_k]
    metric = f"warm recall@{primary_k}"
    floor = baseline.warm.recall[primary_k]
    if score <= floor:
        return False, f"{metric} {score:.4f} does not beat the popularity baseline {floor:.4f}"
    if champion is None:
        return True, f"no champion yet; {metric} {score:.4f} > popularity {floor:.4f}"
    best = champion.warm.recall[primary_k]
    if score > best + min_improvement:
        return True, f"{metric} {score:.4f} > champion {best:.4f} (+{min_improvement})"
    return False, f"{metric} {score:.4f} does not beat champion {best:.4f} (+{min_improvement})"


def run_promotion(
    settings: TrainingSettings, source: TableSource, store: ArtifactStore
) -> EvaluationReport:
    candidate, candidate_meta = store.load_model(settings.model_id)
    champion, champion_meta = _load_champion(store, settings.model_id)
    now = datetime.now(UTC)

    if champion_meta is not None and champion_meta.model_id == settings.model_id:
        report = store.read_report(CHAMPION)
        if report is None:
            raise FileNotFoundError("champion model without evaluation/champion/metrics.json")
        report = report.model_copy(
            update={"evaluated_at": now, "promoted": False, "reason": "already the champion"}
        )
        store.write_report(report, model_id=settings.model_id)
        return report

    # Rows after both training cutoffs: neither model trained on them.
    cutoffs = [candidate_meta.split.cutoff]
    if champion is not None and champion_meta is not None:
        cutoffs.append(champion_meta.split.cutoff)
    device = resolve_device(settings.device)
    data = _load(settings, source, cutoff=max(cutoffs), with_train_rows=False)
    if len(data.validation) == 0:
        raise ValueError(f"no positive interactions after {max(cutoffs)} to evaluate on")

    models = {"candidate": candidate}
    if champion is not None:
        models["champion"] = champion
    results, baseline = _evaluate_all(settings, data, models, device)
    candidate_metrics = results["candidate"]
    champion_eval: ModelEvaluation | None = None
    if champion is not None and champion_meta is not None:
        champion_eval = ModelEvaluation(
            model_id=champion_meta.model_id, metrics=results["champion"]
        )
    elif champion_meta is not None:
        # incompatible champion: fall back to the metrics it was promoted with
        stored = store.read_report(CHAMPION)
        if stored is not None:
            champion_eval = stored.candidate

    promoted, reason = decide(
        settings.primary_k,
        candidate_metrics,
        champion_eval.metrics if champion_eval else None,
        baseline,
        settings.min_improvement,
    )
    if not promoted and settings.force_promotion:
        log.warning("FORCE_PROMOTION: promoting %s although %s", settings.model_id, reason)
        promoted, reason = True, f"forced (FORCE_PROMOTION): {reason}"
    report = EvaluationReport(
        model_id=settings.model_id,
        evaluated_at=now,
        primary_k=settings.primary_k,
        split=data.split,
        candidate=ModelEvaluation(model_id=settings.model_id, metrics=candidate_metrics),
        champion=champion_eval,
        popularity_baseline=baseline,
        promoted=promoted,
        reason=reason,
    )
    compared = {"candidate": candidate_metrics}
    if champion_eval is not None:
        compared["champion"] = champion_eval.metrics
    compared["popularity"] = baseline
    log.info("metrics %s", metrics_line(settings.primary_k, compared))
    log.info("promotion: %s (%s)", "PROMOTED" if promoted else "kept champion", reason)
    store.write_report(report)
    if promoted:
        store.promote(report)
    return report
