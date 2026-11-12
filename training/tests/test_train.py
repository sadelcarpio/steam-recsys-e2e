import io

import numpy as np
import pytest
import torch

from steam_training.batching import NO_GAME, Batch, HardNegatives, build_examples, to_item_batch
from steam_training.evaluation import evaluate, model_scores, popularity_scores
from steam_training.model import TwoTowerModel
from steam_training.train import fingerprint, model_config, retrieval_loss, train_model
from tests.test_data import load
from tests.test_model import config, items


def _batch(targets, history, negatives=()):
    t = np.array(targets)
    return Batch(
        history=torch.tensor(history),
        items=to_item_batch(items(targets)),
        target=torch.from_numpy(t),
        target_log_prob=torch.zeros(len(t)),
        negatives=list(negatives),
    )


def test_loss_masks_duplicate_targets_and_history_items():
    model = TwoTowerModel(config())
    model.sync_user_table()
    # rows 0 and 1 share target 2; row 2's target (4) is in row 0's history
    batch = _batch([2, 2, 4], [[4, 0, 0, 0, 0], [3, 0, 0, 0, 0], [5, 0, 0, 0, 0]])
    loss = retrieval_loss(model, batch, logq_correction=False)
    users = model.user_tower(batch.history)
    logits = model.score(users, model.item_tower(batch.items))
    # expected: row 0 only sees itself, row 1 sees itself + row 2, row 2 sees all
    expected = torch.stack(
        [
            torch.logsumexp(logits[0, [0]], 0) - logits[0, 0],
            torch.logsumexp(logits[1, [1, 2]], 0) - logits[1, 1],
            torch.logsumexp(logits[2, [0, 1, 2]], 0) - logits[2, 2],
        ]
    ).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_invalid_hard_negatives_do_not_change_the_loss():
    model = TwoTowerModel(config())
    model.sync_user_table()
    history = [[3, 0, 0, 0, 0], [5, 0, 0, 0, 0]]
    plain = retrieval_loss(model, _batch([2, 4], history), False)
    invalid = HardNegatives(
        items=to_item_batch(items([6, 7])),
        game_idx=torch.tensor([-1, -1]),
        valid=torch.tensor([False, False]),
    )
    assert torch.allclose(plain, retrieval_loss(model, _batch([2, 4], history, [invalid]), False))
    valid = HardNegatives(invalid.items, torch.tensor([6, 7]), torch.tensor([True, True]))
    assert retrieval_loss(model, _batch([2, 4], history, [valid]), False) > plain


def _examples(source, settings):
    data = load(source, cold_row_fraction=settings.cold_row_fraction)
    return data, build_examples(data.train, data.catalog, data.negatives)


def test_training_learns_user_taste(source, settings):
    settings = settings.model_copy(update={"early_stopping_patience": 0})
    data, examples = _examples(source, settings)
    model, losses = train_model(
        settings, examples, data.catalog, data.vocab, data.validation.sample(100)
    )
    assert model.config == model_config(settings, data.vocab)
    assert len(losses) == settings.epochs and losses[-1] < losses[0]
    assert (examples.mined >= 2).all()  # mined from epoch 1 on
    ks = [5]
    learned = evaluate(
        model_scores(model, data.catalog, 128), data.validation, data.catalog, ks, 128
    )
    popular = evaluate(
        popularity_scores(data.train_positive_counts, data.catalog),
        data.validation,
        data.catalog,
        ks,
        128,
    )
    # 40 games, 10 per taste cluster: random recall@5 is ~0.13
    assert learned.warm.recall[5] > popular.warm.recall[5] + 0.1


def test_mining_is_capped_per_epoch(source, settings):
    data, examples = _examples(source, settings)
    capped = settings.model_copy(update={"epochs": 2, "mine_max_rows": 50})
    train_model(capped, examples, data.catalog, data.vocab)
    assert (examples.mined != NO_GAME).sum() == 50


class MemoryCheckpoints:
    def __init__(self, fail_after_epoch: int | None = None) -> None:
        self.state = None
        self.saves = 0
        self.fail_after_epoch = fail_after_epoch

    def load(self):
        return self.state

    def save(self, state):
        buffer = io.BytesIO()
        torch.save(state, buffer)  # must be serialisable like the S3 version
        self.state = torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
        self.saves += 1
        if self.fail_after_epoch is not None and state["epoch"] == self.fail_after_epoch:
            raise KeyboardInterrupt("simulated crash")


def test_resumed_training_equals_an_uninterrupted_run(source, settings):
    data, _ = _examples(source, settings)
    straight = MemoryCheckpoints()
    _, examples = _examples(source, settings)
    model, losses = train_model(
        settings, examples, data.catalog, data.vocab, checkpoints=straight, run_fingerprint="a"
    )
    assert straight.saves == settings.epochs

    crashing = MemoryCheckpoints(fail_after_epoch=2)
    _, examples = _examples(source, settings)
    with pytest.raises(KeyboardInterrupt):
        train_model(
            settings, examples, data.catalog, data.vocab, checkpoints=crashing, run_fingerprint="a"
        )
    crashing.fail_after_epoch = None
    _, examples = _examples(source, settings)
    resumed, resumed_losses = train_model(
        settings, examples, data.catalog, data.vocab, checkpoints=crashing, run_fingerprint="a"
    )
    assert crashing.saves == settings.epochs  # epochs 1-2, then 3-4 after the resume
    assert resumed_losses == losses
    for (name, a), (_, b) in zip(
        model.state_dict().items(), resumed.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_extended_training_equals_a_longer_run(source, settings):
    longer = MemoryCheckpoints()
    data, examples = _examples(source, settings)
    model, losses = train_model(
        settings, examples, data.catalog, data.vocab, checkpoints=longer, run_fingerprint="a"
    )

    # a finished 2-epoch run, then the same run with EPOCHS=4
    extended = MemoryCheckpoints()
    short = settings.model_copy(update={"epochs": 2})
    _, examples = _examples(source, settings)
    train_model(
        short, examples, data.catalog, data.vocab, checkpoints=extended, run_fingerprint="a"
    )
    _, examples = _examples(source, settings)
    resumed, resumed_losses = train_model(
        settings, examples, data.catalog, data.vocab, checkpoints=extended, run_fingerprint="a"
    )
    assert extended.saves == settings.epochs  # epochs 1-2, then only 3-4
    assert resumed_losses == losses
    for (name, a), (_, b) in zip(
        model.state_dict().items(), resumed.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_checkpoint_past_epochs_is_ignored(source, settings):
    checkpoints = MemoryCheckpoints()
    data, examples = _examples(source, settings)
    train_model(settings, examples, data.catalog, data.vocab, checkpoints=checkpoints)
    fewer = settings.model_copy(update={"epochs": 1})
    _, examples = _examples(source, settings)
    _, losses = train_model(fewer, examples, data.catalog, data.vocab, checkpoints=checkpoints)
    assert len(losses) == 1  # started over instead of returning the 4-epoch model


def test_checkpoint_of_another_run_is_ignored(source, settings):
    data, examples = _examples(source, settings)
    other = MemoryCheckpoints()
    one = settings.model_copy(update={"epochs": 1})
    train_model(one, examples, data.catalog, data.vocab, checkpoints=other, run_fingerprint="x")
    _, losses = train_model(
        one, examples, data.catalog, data.vocab, checkpoints=other, run_fingerprint="y"
    )
    assert len(losses) == 1  # trained from scratch instead of resuming a finished run


def test_fingerprint_ignores_evaluation_settings(settings):
    base = fingerprint(settings, {"interactions": 1}, "2024-01-01")
    assert (
        fingerprint(
            settings.model_copy(update={"epochs": 99, "primary_k": 10}),
            {"interactions": 1},
            "2024-01-01",
        )
        == base
    )
    assert (
        fingerprint(
            settings.model_copy(update={"learning_rate": 0.1}), {"interactions": 1}, "2024-01-01"
        )
        != base
    )
    assert fingerprint(settings, {"interactions": 2}, "2024-01-01") != base


def test_in_epoch_table_sync(source, settings, monkeypatch):
    """USER_TABLE_SYNC_STEPS re-syncs the frozen game table every N steps, never after the
    last step; resuming still reproduces an uninterrupted run."""
    synced = settings.model_copy(update={"epochs": 2, "user_table_sync_steps": 3})
    calls = []
    original = TwoTowerModel.sync_user_table

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(TwoTowerModel, "sync_user_table", counting)
    data, examples = _examples(source, synced)
    model, losses = train_model(synced, examples, data.catalog, data.vocab)
    steps = (len(examples) + synced.batch_size - 1) // synced.batch_size
    if len(examples) > synced.batch_size:
        steps = len(examples) // synced.batch_size  # drop_last
    per_epoch = 1 + (steps - 1) // 3  # start of epoch + every 3 steps before a step
    assert len(calls) == 2 * per_epoch
    # the user tower's table is the item table as of the last sync, not the final one
    assert not torch.equal(
        model.user_tower.game_table, model.item_tower.game_embedding.weight.detach()
    )
    assert fingerprint(synced, {}, "c") != fingerprint(settings, {}, "c")


def _scripted_monitor(monkeypatch, scores):
    """Per-epoch monitor warm recall@PRIMARY_K taken from `scores`, in order."""
    from steam_training.contracts import RecallMetrics, SegmentRecall

    remaining = list(scores)

    def fake_evaluate(score_fn, rows, catalog, ks, batch_size, device=None):
        segment = SegmentRecall(rows=1, recall={k: remaining[0] for k in ks})
        remaining.pop(0)
        return RecallMetrics(warm=segment, cold=segment, all=segment)

    monkeypatch.setattr("steam_training.train.evaluate", fake_evaluate)


def _same_weights(a, b) -> bool:
    return all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


def test_keeps_the_best_epoch(source, settings, monkeypatch):
    base = settings.model_copy(update={"early_stopping_patience": 0})
    data, examples = _examples(source, base)
    rows = data.validation.sample(20)
    _scripted_monitor(monkeypatch, [0.1, 0.3, 0.2, 0.25])
    result = train_model(
        base.model_copy(update={"epochs": 4}), examples, data.catalog, data.vocab, rows
    )
    assert (result.best_epoch, result.epochs_trained) == (2, 4)
    # the same weights as a run that stops after epoch 2 (deterministic training)
    _scripted_monitor(monkeypatch, [0.1, 0.3])
    _, examples = _examples(source, base)
    two = train_model(
        base.model_copy(update={"epochs": 2}), examples, data.catalog, data.vocab, rows
    )
    assert _same_weights(result[0], two[0])
    # KEEP_BEST_EPOCH=false: the last epoch
    _scripted_monitor(monkeypatch, [0.1, 0.3, 0.2, 0.25])
    _, examples = _examples(source, base)
    last = train_model(
        base.model_copy(update={"epochs": 4, "keep_best_epoch": False}),
        examples,
        data.catalog,
        data.vocab,
        rows,
    )
    assert last.best_epoch == 4 and not _same_weights(last[0], result[0])


def test_early_stopping(source, settings, monkeypatch):
    patient = settings.model_copy(update={"epochs": 6, "early_stopping_patience": 2})
    data, examples = _examples(source, patient)
    _scripted_monitor(monkeypatch, [0.1, 0.3, 0.2, 0.25, 0.9, 0.9])
    result = train_model(patient, examples, data.catalog, data.vocab, data.validation.sample(20))
    # epochs 3 and 4 don't beat epoch 2: stop before epoch 5
    assert (result.best_epoch, result.epochs_trained, len(result[1])) == (2, 4, 4)


def test_no_monitoring_keeps_the_last_epoch(source, settings):
    data, examples = _examples(source, settings)
    result = train_model(settings, examples, data.catalog, data.vocab)
    assert (result.best_epoch, result.epochs_trained) == (settings.epochs, settings.epochs)


def test_best_epoch_survives_a_resume(source, settings, monkeypatch):
    base = settings.model_copy(update={"epochs": 4, "early_stopping_patience": 0})
    data, _ = _examples(source, base)
    rows = data.validation.sample(20)
    scores = [0.1, 0.3, 0.2, 0.25]
    _scripted_monitor(monkeypatch, scores)
    _, examples = _examples(source, base)
    straight = train_model(base, examples, data.catalog, data.vocab, rows)

    crashing = MemoryCheckpoints(fail_after_epoch=3)  # best (epoch 2) is behind the checkpoint
    _scripted_monitor(monkeypatch, scores[:3])
    _, examples = _examples(source, base)
    with pytest.raises(KeyboardInterrupt):
        train_model(
            base,
            examples,
            data.catalog,
            data.vocab,
            rows,
            checkpoints=crashing,
            run_fingerprint="a",
        )
    assert crashing.state["best"]["epoch"] == 2 and crashing.state["best"]["model"] is not None
    crashing.fail_after_epoch = None
    _scripted_monitor(monkeypatch, scores[3:])
    _, examples = _examples(source, base)
    resumed = train_model(
        base, examples, data.catalog, data.vocab, rows, checkpoints=crashing, run_fingerprint="a"
    )
    assert (resumed.best_epoch, resumed.epochs_trained) == (2, 4)
    assert _same_weights(resumed[0], straight[0])
