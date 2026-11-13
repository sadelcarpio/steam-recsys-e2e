"""Stage 3, training: sampled softmax over in-batch + hard negatives.

For each example (user query u_i, target item v_i) the logits are
    u_i . v_j / T - log q(j)   for every batch target j (in-batch negatives, logQ-corrected),
    u_i . h_i / T              for each of its hard negatives (explicit, mined),
with collisions masked: another example's target that is the same game as v_i, or a game in
u_i's history, is not a negative. Cross-entropy picks v_i.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

from steam_training.batching import (
    Batch,
    Collator,
    TrainingExamples,
    make_loader,
    target_log_probs,
)
from steam_training.config import NOT_FINGERPRINTED, TrainingSettings
from steam_training.contracts import EpochMetrics, ModelConfig, VocabSizes
from steam_training.data import Catalog, EvalRows
from steam_training.evaluation import (
    evaluate,
    mine_hard_negatives,
    model_scores,
)
from steam_training.model import TwoTowerModel

log = logging.getLogger(__name__)


def model_config(settings: TrainingSettings, vocab: VocabSizes) -> ModelConfig:
    return ModelConfig(
        vocab=vocab,
        game_embedding_dim=settings.game_embedding_dim,
        attribute_embedding_dim=settings.attribute_embedding_dim,
        hidden_dim=settings.hidden_dim,
        output_dim=settings.output_dim,
        temperature=settings.temperature,
        user_tag_profile=settings.user_tag_profile,
        history_length=settings.history_length,
        history_encoder=settings.history_encoder,
        attention_heads=settings.attention_heads,
        attention_layers=settings.attention_layers,
    )


def retrieval_loss(model: TwoTowerModel, batch: Batch, logq_correction: bool) -> torch.Tensor:
    users = model.user_tower(batch.history)
    items = model.item_tower(batch.items)
    logits = model.score(users, items)
    if logq_correction:
        logits = logits - batch.target_log_prob[None, :]
    target = batch.target
    collision = target[None, :] == target[:, None]
    # one history slot at a time: [B, B] instead of [B, history_length, B] (A3: 50+ slots)
    in_history = torch.zeros_like(collision)
    for slot in range(batch.history.shape[1]):
        in_history |= batch.history[:, slot, None] == target[None, :]
    diagonal = torch.eye(len(batch), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill((collision | in_history) & ~diagonal, float("-inf"))

    columns = [logits]
    for negatives in batch.negatives:
        negative_items = model.item_tower(negatives.items)
        hard = (users * negative_items).sum(dim=-1) / model.config.temperature
        columns.append(hard.masked_fill(~negatives.valid, float("-inf")).unsqueeze(1))
    labels = torch.arange(len(batch), device=logits.device)
    return F.cross_entropy(torch.cat(columns, dim=1), labels)


class Checkpoints(Protocol):
    """Where per-epoch training state goes (S3 in jobs: `artifacts.S3Checkpoints`)."""

    def load(self) -> dict | None: ...

    def save(self, state: dict) -> None: ...


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but CUDA is not available (CPU torch build or no GPU)")
    return torch.device(setting)


def fingerprint(settings: TrainingSettings, snapshots: dict, cutoff: object) -> str:
    """Identity of a training run's data + hyperparameters: a checkpoint is only resumed into a
    run with the same fingerprint."""
    payload = {
        "settings": settings.model_dump(mode="json", exclude=NOT_FINGERPRINTED),
        "snapshots": snapshots,
        "cutoff": str(cutoff),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _epoch_seed(seed: int, epoch: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, epoch, stream]).generate_state(1)[0])


def _mine(
    settings: TrainingSettings,
    model: TwoTowerModel,
    examples: TrainingExamples,
    catalog: Catalog,
    epoch: int,
    device: torch.device,
) -> None:
    """Re-mine up to MINE_MAX_ROWS random examples; the others keep their previous negative."""
    rng = np.random.default_rng(_epoch_seed(settings.seed, epoch, 1))
    if len(examples) > settings.mine_max_rows:
        rows = np.sort(rng.choice(len(examples), settings.mine_max_rows, replace=False))
    else:
        rows = np.arange(len(examples))
    examples.mined[rows] = mine_hard_negatives(
        model,
        examples.history[rows],
        examples.target[rows],
        catalog,
        skip_top=settings.mine_skip_top,
        pool_size=settings.mine_pool_size,
        generator=torch.Generator().manual_seed(_epoch_seed(settings.seed, epoch, 2)),
        batch_size=settings.eval_batch_size,
        device=device,
    )


class TrainResult(tuple):
    """`(model, epoch_losses)`, unpacked like before, plus `best_epoch` (1-based: the epoch whose
    weights the model has) and `epochs_trained` (fewer than EPOCHS after an early stop)."""

    best_epoch: int
    epochs_trained: int

    def __new__(cls, model, epoch_losses, best_epoch, epochs_trained):
        result = super().__new__(cls, (model, epoch_losses))
        result.best_epoch = best_epoch
        result.epochs_trained = epochs_trained
        return result


def _cpu_state(model: TwoTowerModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(
    settings: TrainingSettings,
    examples: TrainingExamples,
    catalog: Catalog,
    vocab: VocabSizes,
    monitor_rows: EvalRows | None = None,
    *,
    device: torch.device | None = None,
    checkpoints: Checkpoints | None = None,
    run_fingerprint: str = "",
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> TrainResult:
    """`on_epoch` gets each epoch's metrics (experiment tracking), including the epochs restored
    from a checkpoint (loss only). With `monitor_rows`, the epoch with the best warm
    recall@PRIMARY_K is tracked: KEEP_BEST_EPOCH returns its weights, and
    EARLY_STOPPING_PATIENCE stops after that many epochs without improvement. Both survive a
    resume (the best weights are in the checkpoint)."""
    device = device or torch.device("cpu")
    torch.manual_seed(settings.seed)
    model = TwoTowerModel(model_config(settings, vocab), settings.attention_dropout)
    seen = torch.zeros(vocab.games, dtype=torch.bool)
    seen[torch.from_numpy(examples.target)] = True
    model.set_seen_games(seen)
    if model.config.user_tag_profile:
        model.set_game_tags(catalog.game_tags(vocab.games))
    model.to(device)

    collator = Collator(
        examples,
        catalog,
        target_log_probs(examples.target, vocab.games),
        history_dropout=settings.history_dropout,
        item_id_dropout=settings.item_id_dropout,
        explicit_negatives=settings.explicit_negatives,
        mined_negatives=settings.mined_negatives,
        seed=settings.seed,
    )
    shuffle = torch.Generator()
    loader = make_loader(examples, collator, settings.batch_size, shuffle)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    epoch_losses: list[float] = []
    first_epoch = 0
    # best monitor score so far: epoch (1-based), score and a CPU copy of its weights
    best_epoch: int | None = None
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    state = checkpoints.load() if checkpoints is not None and settings.resume else None
    if state is not None and int(state["epoch"]) > settings.epochs:
        # Resuming would return a model trained longer than EPOCHS says.
        log.warning(
            "ignoring a checkpoint after epoch %d: EPOCHS=%d is lower",
            state["epoch"],
            settings.epochs,
        )
        state = None
    if state is not None and state["fingerprint"] == run_fingerprint:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        examples.mined[:] = state["mined"].numpy()
        epoch_losses = list(state["epoch_losses"])
        first_epoch = int(state["epoch"])
        best = state.get("best")
        if best is not None:
            best_epoch, best_score = int(best["epoch"]), float(best["score"])
            best_state = best["model"] if best["model"] is not None else state["model"]
        log.info("resumed from the checkpoint after epoch %d", first_epoch)
        if on_epoch is not None:
            for number, loss in enumerate(epoch_losses, start=1):
                on_epoch(EpochMetrics(epoch=number, loss=loss))
    elif state is not None:
        log.warning("ignoring a checkpoint from a different run (data or hyperparameters changed)")
    log.info(
        "training on %d examples, %d catalog games, %d batches/epoch, device %s",
        len(examples),
        len(catalog),
        len(loader),
        device,
    )

    epochs_trained = first_epoch
    for epoch in range(first_epoch, settings.epochs):
        patience = settings.early_stopping_patience
        if patience and best_epoch is not None and epoch - best_epoch >= patience:
            log.info(
                "early stopping: no better warm_recall@%d in the %d epoch(s) after epoch %d",
                settings.primary_k,
                epoch - best_epoch,
                best_epoch,
            )
            break
        started = time.monotonic()
        # The user tower reads a snapshot of the game table that only moves between epochs.
        model.sync_user_table()
        if settings.mined_negatives and epoch >= settings.mining_start_epoch:
            _mine(settings, model, examples, catalog, epoch, device)
        collator.reseed(epoch)
        shuffle.manual_seed(_epoch_seed(settings.seed, epoch, 0))
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            # before a step, never after the last one: the user MLP always trains against the
            # table it ends with
            sync_every = settings.user_table_sync_steps
            if sync_every and batches and batches % sync_every == 0:
                model.sync_user_table()
            loss = retrieval_loss(model, batch.to(device), settings.logq_correction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        epoch_losses.append(total / max(batches, 1))
        message = f"epoch {epoch + 1}/{settings.epochs} loss={epoch_losses[-1]:.4f}"
        monitor: float | None = None
        if monitor_rows is not None and len(monitor_rows):
            metrics = evaluate(
                model_scores(model, catalog, settings.eval_batch_size, device),
                monitor_rows,
                catalog,
                [settings.primary_k],
                settings.eval_batch_size,
                device,
            )
            monitor = metrics.warm.recall[settings.primary_k]
            message += f" warm_recall@{settings.primary_k}={monitor:.4f}"
        if monitor is not None and monitor > best_score:
            best_epoch, best_score, best_state = epoch + 1, monitor, _cpu_state(model)
        seconds = time.monotonic() - started
        log.info("%s (%.0fs)", message, seconds)
        if on_epoch is not None:
            on_epoch(
                EpochMetrics(
                    epoch=epoch + 1,
                    loss=epoch_losses[-1],
                    monitor_warm_recall=monitor,
                    seconds=seconds,
                )
            )
        epochs_trained = epoch + 1
        if checkpoints is not None:
            best = None
            if best_epoch is not None:
                # the best weights are only stored when they aren't this checkpoint's model
                stored = best_state if best_epoch != epoch + 1 else None
                best = {"epoch": best_epoch, "score": best_score, "model": stored}
            checkpoints.save(
                {
                    "fingerprint": run_fingerprint,
                    "epoch": epoch + 1,
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "mined": torch.from_numpy(examples.mined.copy()),
                    "epoch_losses": epoch_losses,
                    "best": best,
                }
            )
    if settings.keep_best_epoch and best_state is not None and best_epoch != epochs_trained:
        log.info(
            "keeping epoch %d (warm_recall@%d=%.4f), not the last one (%d)",
            best_epoch,
            settings.primary_k,
            best_score,
            epochs_trained,
        )
        model.load_state_dict(best_state)
    kept = best_epoch if settings.keep_best_epoch and best_state is not None else epochs_trained
    model.eval()
    return TrainResult(model, epoch_losses, kept or epochs_trained, epochs_trained)
