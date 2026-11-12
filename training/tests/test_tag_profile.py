"""A4 (spec 17): the user tower's tag profile of the history."""

import io

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws
from pydantic import ValidationError

from steam_training.artifacts import USER_TOWER_NUMPY, ArtifactStore, model_prefix
from steam_training.config import TrainingSettings
from steam_training.contracts import ModelConfig, VocabSizes
from steam_training.data import Ragged
from steam_training.export import TAG_PROFILE_ARRAYS, USER_TOWER_TAG_PROFILE_FORMAT
from steam_training.model import TwoTowerModel
from steam_training.pipeline import run_promotion, run_training
from tests.conftest import BUCKET

VOCAB = VocabSizes(games=8, developers=3, publishers=3, genres=3, categories=3, tags=6)


def _model(profile: bool = True) -> TwoTowerModel:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab=VOCAB,
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=8,
        output_dim=8,
        temperature=0.05,
        user_tag_profile=profile,
    )
    model = TwoTowerModel(config)
    # games 2, 3: tag 2; game 4: tag 3 (+ a little tag 4); games 5-7 no tags
    tags = Ragged.from_lists([[], [], [2], [2], [3, 4], [], [], []])
    weights = np.array([1.0, 1.0, 0.9, 0.1], dtype=np.float32)
    if profile:
        model.set_game_tags(Ragged(tags.values, tags.offsets, weights))
    seen = torch.ones(VOCAB.games, dtype=torch.bool)
    seen[3] = False  # game 3: no training interactions (OOV in the game table)
    model.set_seen_games(seen)
    model.eval()
    return model


def test_profile_is_a_normalized_weighted_bag_of_the_history_tags():
    model = _model()
    tower = model.user_tower
    history = torch.tensor([[2, 4, 0, 0, 0], [0, 0, 0, 0, 0], [5, 6, 0, 0, 0]])
    profile = tower._tag_profile(history)
    table = tower.tag_profile.weight
    # equal position weights at init: game 2 (tag 2: 1.0) + game 4 (tags 3, 4: 0.9, 0.1)
    expected = (table[2] * 1.0 + table[3] * 0.9 + table[4] * 0.1) / 2.0
    assert torch.allclose(profile[0], expected, atol=1e-6)
    assert torch.equal(profile[1], torch.zeros(4))  # empty history
    assert torch.equal(profile[2], torch.zeros(4))  # games without tags


def test_unseen_games_still_contribute_their_tags():
    model = _model()
    tower = model.user_tower
    # game 3 is OOV for the game table but has tag 2, like game 2
    seen_tags = tower._tag_profile(torch.tensor([[2, 0, 0, 0, 0]]))
    unseen_tags = tower._tag_profile(torch.tensor([[3, 0, 0, 0, 0]]))
    assert torch.allclose(seen_tags, unseen_tags)
    # ids beyond the vocabulary have no tags
    assert torch.equal(tower._tag_profile(torch.tensor([[50, 0, 0, 0, 0]])), torch.zeros(1, 4))


def test_profile_changes_the_user_embedding_and_is_trainable():
    with_profile, without = _model(True), _model(False)
    history = torch.tensor([[2, 4, 0, 0, 0]])
    assert with_profile.user_tower(history).shape == without.user_tower(history).shape
    with_profile.train()
    with_profile.user_tower(history).sum().backward()
    assert with_profile.user_tower.tag_recency.grad is not None
    assert with_profile.user_tower.tag_profile.weight.grad is not None


def test_settings_need_the_tags():
    with pytest.raises(ValidationError):
        TrainingSettings(
            model_artifacts_bucket=BUCKET,
            model_id="m",
            use_game_tags=False,
            user_tag_profile=True,
        )


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_train_save_reload_export_and_promote(settings, source, store):
    settings = settings.model_copy(update={"user_tag_profile": True})
    metadata = run_training(settings, source, store)
    assert metadata.config.user_tag_profile
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    # reload: the tag lookup comes back with its saved size
    model, _ = store.load_model(settings.model_id)
    assert len(model.user_tower.game_tag_values) > 0
    history = torch.tensor([[2, 3, 4, 0, 0]])
    assert torch.isfinite(model.user_tower(history)).all()
    # the numpy user tower (format 2) is saved, so the model can be promoted
    body = store._get(model_prefix(settings.model_id) + USER_TOWER_NUMPY)
    with np.load(io.BytesIO(body)) as npz:
        assert int(npz["format_version"]) == USER_TOWER_TAG_PROFILE_FORMAT
        assert set(TAG_PROFILE_ARRAYS) <= set(npz.files)
        assert np.array_equal(npz["tag_offsets"], model.user_tower.game_tag_offsets.numpy())
        assert npz["tag_values"].dtype == np.uint16
        assert np.array_equal(npz["tag_values"], model.user_tower.game_tag_values.numpy())
    report = run_promotion(settings, source, store)
    assert report.promoted


def test_catalog_game_tags_by_game_idx(source):
    from steam_training.data import load_training_data

    data = load_training_data(
        source,
        validation_fraction=0.1,
        cold_row_fraction=0.0,
        eval_max_rows=10**9,
        seed=0,
        with_tags=True,
    )
    tags = data.catalog.game_tags(data.vocab.games)
    assert len(tags) == data.vocab.games
    assert tags.row(0) == [] and tags.row(1) == []  # padding, OOV
    for game in data.catalog.items.game_idx[:5].tolist():
        row = int(data.catalog.row_of[game])
        assert tags.row(game) == data.catalog.items.tags.row(row)
    for game in range(data.vocab.games):
        lo, hi = tags.offsets[game], tags.offsets[game + 1]
        if hi > lo:
            assert tags.weights[lo:hi].sum() == pytest.approx(1.0, abs=1e-5)
