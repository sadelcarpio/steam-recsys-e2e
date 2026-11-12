"""A3 (spec 17): histories longer than the mart's 5 games, derived from `interactions`, and the
self-attention history encoder."""

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws
from pydantic import ValidationError

from steam_training.artifacts import ArtifactStore, model_prefix
from steam_training.config import TrainingSettings
from steam_training.contracts import ModelConfig, VocabSizes
from steam_training.data import HistoryIndex, build_history_index, load_training_data
from steam_training.model import HistoryAttention, TwoTowerModel
from steam_training.pipeline import run_promotion, run_training
from tests.conftest import BUCKET, FakeSource, make_marts


def test_history_index_takes_the_last_positives_strictly_before_each_row():
    # user 7: games 2, 3, 4 at t = 10, 20, 30; user 9: game 5 at t = 10
    user = np.array([7, 9, 7, 7])
    ts = np.array([20, 10, 10, 30])
    game = np.array([3, 5, 2, 4])
    rows_user = np.array([7, 7, 7, 9, 8])
    rows_ts = np.array([30, 31, 10, 11, 50])
    index = build_history_index(user, ts, game, rows_user, rows_ts, length=2)
    assert index.shape == (5, 2)
    assert index[np.arange(5)].tolist() == [
        [3, 2],  # before t=30: the review at 30 itself is not in its own history
        [4, 3],  # truncated to the 2 most recent
        [0, 0],  # nothing strictly before t=10
        [5, 0],
        [0, 0],  # unknown user
    ]
    taken = index.take(np.array([1, 3]))
    assert isinstance(taken, HistoryIndex)
    assert taken.matrix().tolist() == [[4, 3], [5, 0]]


def _load(source, length, **kwargs):
    return load_training_data(
        source,
        validation_fraction=0.1,
        cold_row_fraction=0.3,
        eval_max_rows=10**9,
        seed=0,
        history_length=length,
        **kwargs,
    )


def test_derived_histories_extend_the_mart_history():
    source = FakeSource(make_marts(n_users=200, reviews_per_user=14))  # up to 10 positives
    mart, long = _load(source, 5), _load(source, 12)
    assert isinstance(long.train.history, HistoryIndex)
    train_long = long.train.history.matrix()
    assert train_long.shape == (len(mart.train), 12)
    # same rows, and the derived history starts with the mart's 5 games
    assert np.array_equal(long.train.target, mart.train.target)
    assert np.array_equal(train_long[:, :5], mart.train.history)
    assert np.array_equal(long.validation.history[:, :5], mart.validation.history)
    assert np.array_equal(long.validation.is_warm, mart.validation.is_warm)
    # and is longer for users with more than 5 earlier positives
    assert (train_long[:, 5] != 0).any()
    assert long.train.timestamp is None  # only kept to build the index


def _model(encoder="attention", length=6):
    torch.manual_seed(0)
    config = ModelConfig(
        vocab=VocabSizes(games=10, developers=3, publishers=3, genres=3, categories=3),
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=8,
        output_dim=8,
        temperature=0.05,
        history_length=length,
        history_encoder=encoder,
    )
    model = TwoTowerModel(config)
    torch.nn.init.normal_(model.item_tower.game_embedding.weight)
    model.sync_user_table()
    return model.eval()


def test_attention_ignores_padding_and_handles_empty_histories():
    attention = HistoryAttention(dim=8, length=4, heads=2, layers=2, dropout=0.0).eval()
    games = torch.randn(3, 4, 8)
    present = torch.tensor([[True, True, False, False], [False] * 4, [True] * 4])
    out = attention(games, present)
    assert out.shape == (3, 8) and torch.isfinite(out).all()
    noisy = games.clone()
    noisy[~present] = torch.randn(int((~present).sum()), 8) * 100
    assert torch.allclose(attention(noisy, present), out, atol=1e-5)
    # order matters (learned positions): the most recent game is position 0
    swapped = games[:, [1, 0, 2, 3]]
    assert not torch.allclose(attention(swapped, present)[0], out[0])


def test_user_tower_reads_its_history_length():
    model = _model()
    history = torch.tensor([[2, 3, 4, 0, 0, 0], [0] * 6])
    users = model.user_tower(history)
    assert users.shape == (2, 8) and torch.isfinite(users).all()
    with pytest.raises(ValueError):
        model.user_tower(history[:, :5])
    model.train()
    model.user_tower(history).sum().backward()
    assert model.user_tower.attention.position.grad is not None
    assert model.user_tower.game_table.grad is None  # still a frozen buffer


def test_settings_check_the_attention_heads():
    with pytest.raises(ValidationError):
        TrainingSettings(
            model_artifacts_bucket=BUCKET, model_id="m", game_embedding_dim=10, attention_heads=3
        )


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_train_save_reload_and_refuse_promotion(settings, source, store):
    settings = settings.model_copy(
        update={
            "history_length": 8,
            "history_encoder": "attention",
            "user_tag_profile": True,
            "game_embedding_dim": 16,
        }
    )
    metadata = run_training(settings, source, store)
    config = metadata.config
    assert (config.history_length, config.history_encoder) == (8, "attention")
    assert not config.servable
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, _ = store.load_model(settings.model_id)
    assert model.user_tower.tag_recency.shape == (8,)
    assert torch.isfinite(model.user_tower(torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]]))).all()
    # no numpy port of attention yet: no user_tower.npz, and promotion refuses the model
    assert not store._exists(model_prefix(settings.model_id) + "user_tower.npz")
    with pytest.raises(ValueError, match="can't be promoted"):
        run_promotion(settings, source, store)
