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

    def delete(self):
        self.state = None


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
