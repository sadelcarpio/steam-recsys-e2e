import numpy as np
import torch

from steam_training.batching import (
    NO_GAME,
    Collator,
    build_examples,
    history_dropout,
    make_loader,
    target_log_probs,
)
from steam_training.contracts import OOV_ID
from tests.test_data import load


def test_explicit_negatives_are_the_users_negative_reviews(source):
    data = load(source)
    examples = build_examples(data.train, data.catalog, data.negatives)
    for i in range(0, len(examples), 97):
        user = data.train.user_id[i]
        expected = set(data.negatives.game_idx[data.negatives.user_id == user])
        start, count = examples.negative_start[i], examples.negative_count[i]
        assert set(examples.negative_games[start : start + count]) == expected
    assert (data.catalog.items.game_idx[examples.catalog_row] == examples.target).all()


def test_history_dropout_keeps_a_non_empty_most_recent_prefix():
    history = np.array([[5, 4, 3, 2, 0], [7, 0, 0, 0, 0], [0, 0, 0, 0, 0]] * 50)
    out = history_dropout(history, 1.0, np.random.default_rng(0))
    lengths = (out != 0).sum(axis=1)
    assert (lengths[0::3] >= 1).all() and (lengths[0::3] <= 3).all()
    for row, original in zip(out, history, strict=True):
        n = (row != 0).sum()
        assert row[:n].tolist() == original[:n].tolist()
    assert (out[1::3] == history[1::3]).all() and (out[2::3] == 0).all()
    assert (history_dropout(history, 0.0, np.random.default_rng(0)) == history).all()


def test_target_log_probs():
    lp = target_log_probs(np.array([2, 2, 3, 2]), 5)
    assert np.isclose(np.exp(lp[2]), 0.75) and np.isclose(np.exp(lp[3]), 0.25)
    assert np.isneginf(lp[4])


def _collator(source, **kwargs):
    data = load(source, cold_row_fraction=0.1)
    examples = build_examples(data.train, data.catalog, data.negatives)
    log_probs = target_log_probs(examples.target, data.vocab.games)
    return examples, Collator(examples, data.catalog, log_probs, seed=0, **kwargs)


def test_collator_builds_consistent_batches(source):
    examples, collator = _collator(source, item_id_dropout=1.0)
    examples.mined[:] = examples.target  # mined == target must be masked out
    batch = next(iter(make_loader(examples, collator, 32, torch.Generator().manual_seed(0))))
    assert len(batch) == 32 and batch.history.shape == (32, 5)
    assert (batch.items.game_idx == OOV_ID).all()  # id dropout replaces the tower input only
    assert (batch.target >= 2).all()
    assert torch.isfinite(batch.target_log_prob).all()
    # numerical features as of the review, list features from the catalog
    assert torch.allclose(batch.items.numeric[:, 2], torch.tensor(0.9))
    explicit, mined = batch.negatives
    assert not mined.valid.any() and (mined.game_idx == NO_GAME).all()
    assert explicit.valid.any()
    assert (explicit.game_idx[explicit.valid] != batch.target[explicit.valid]).all()
    assert len(explicit.items) == 32
    moved = batch.to(torch.device("cpu"))
    assert torch.equal(moved.target, batch.target)


def test_reseed_makes_batches_reproducible(source):
    examples, collator = _collator(source, history_dropout=0.5, item_id_dropout=0.5)
    idx = list(range(64))
    collator.reseed(3)
    a = collator(idx)
    collator(idx)
    collator.reseed(3)
    b = collator(idx)
    assert torch.equal(a.history, b.history) and torch.equal(a.items.game_idx, b.items.game_idx)
