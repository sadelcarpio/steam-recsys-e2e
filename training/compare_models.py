"""Read-only side by side: score several saved models on the same validation rows (after every
model's cutoff, so none trained on them), like promotion does, without writing anything.
Usage (worktree training/): PYTHONPATH=src <venv>/python compare_models.py champion <id> ...
Needs the same env as a training run (AWS_PROFILE, MODEL_ARTIFACTS_BUCKET, DATA_CACHE_DIR).
With an A6 model (item recency) it also scores the weekly recent-popularity baseline, and
prints recall@50 by activity bucket for every row of the table."""

import logging
import sys

from steam_training.artifacts import ArtifactStore
from steam_training.config import TrainingSettings
from steam_training.data import IcebergSource
from steam_training.diagnostics import retrieval_diagnostics
from steam_training.evaluation import (
    model_scores,
    popularity_scores,
    rank_rows,
    recency_rescore,
    recent_popularity_rescore,
    recent_popularity_scores,
)
from steam_training.languages import load_review_languages
from steam_training.pipeline import _load
from steam_training.train import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
ids = sys.argv[1:] or ["champion"]
settings = TrainingSettings(model_id="compare")
store = ArtifactStore(settings.model_artifacts_bucket)
source = IcebergSource(settings.glue_database, settings.aws_region, settings.data_cache_dir)
loaded = {i: store.load_model(i) for i in ids}
models = {i: model for i, (model, _) in loaded.items()}
metas = {i: meta for i, (_, meta) in loaded.items()}
cutoff = max(meta.split.cutoff for meta in metas.values())
data = _load(
    settings,
    source,
    cutoff=cutoff,
    with_train_rows=False,
    with_tags=any(m.config.vocab.tags is not None for m in models.values()),
    history_length=max(m.config.history_length for m in models.values()),
    # A6 models read item recency (others ignore the extra columns)
    with_recency=any(m.config.item_recency for m in models.values()),
    # A7 models read the user's language (`review_languages`); others ignore it
    languages=load_review_languages(source)
    if any(m.config.user_language for m in models.values())
    else None,
)
device = resolve_device(settings.device)
ks = settings.recall_ks


def diagnose(score_fn, rescore=None):
    ranks, exposure = rank_rows(
        score_fn, data.validation, data.catalog, max(ks), settings.eval_batch_size, device, rescore
    )
    return retrieval_diagnostics(ranks, exposure, data.validation, ks)


results = {}
for name, model in models.items():
    model = model.to(device).eval()
    results[name] = diagnose(
        model_scores(model, data.catalog, settings.eval_batch_size, device),
        recency_rescore(model, settings.eval_batch_size, device),
    )
results["popularity"] = diagnose(
    popularity_scores(data.train_positive_counts, data.catalog, device)
)
if data.catalog.recency is not None:
    results["recent popularity (weekly)"] = diagnose(
        recent_popularity_scores(data.catalog, device), recent_popularity_rescore(device)
    )

print(f"\nrows after {cutoff} (every model's cutoff); trained on / stored warm recall@50:")
for i in ids:
    meta = metas[i]
    stored = meta.validation.warm.recall[50]
    print(f"  {i}: {meta.model_id}, cutoff {meta.split.cutoff}, stored {stored:.4f}")
print(f"\n{'model':45s} {'warm@30':>8s} {'warm@50':>8s} {'warm@100':>8s} {'all@50':>8s}")
for name, diag in results.items():
    w = diag.recall.warm.recall
    print(f"{name:45s} {w[30]:8.4f} {w[50]:8.4f} {w[100]:8.4f} {diag.recall.all.recall[50]:8.4f}")
buckets = [b.name for b in next(iter(results.values())).by_activity]
print(f"\nrecall@50 by activity\n{'model':45s} " + " ".join(f"{b:>8s}" for b in buckets))
for name, diag in results.items():
    print(f"{name:45s} " + " ".join(f"{b.recall[50]:8.4f}" for b in diag.by_activity))
