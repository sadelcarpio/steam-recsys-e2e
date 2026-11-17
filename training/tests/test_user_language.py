"""A7 (spec 17): the language of the user's previous review as a user tower input, read from the
raw reviews (`rec_id` = the marts' `review_id`)."""

import boto3
import numpy as np
import pyarrow as pa
import pytest
import torch
from moto import mock_aws
from pydantic import ValidationError

from steam_training import evaluation
from steam_training.artifacts import USER_TOWER_NUMPY, ArtifactStore, model_prefix
from steam_training.batching import LANGUAGE_DROPOUT, Collator, build_examples, target_log_probs
from steam_training.config import TrainingSettings
from steam_training.contracts import (
    FIRST_ID,
    NUM_LANGUAGE_IDS,
    OOV_ID,
    STEAM_LANGUAGES,
    ModelConfig,
    VocabSizes,
)
from steam_training.data import load_training_data, previous_language
from steam_training.languages import ReviewLanguages, language_codes
from steam_training.model import TwoTowerModel
from steam_training.pipeline import run_promotion, run_training
from tests.conftest import BUCKET, FakeSource, make_marts

LANGS = ("english", "schinese", "russian", None)


def test_language_codes_and_lookup():
    codes = language_codes(pa.array(["english", "Russian", "klingon", None, "ukrainian"]))
    english, russian = FIRST_ID, FIRST_ID + STEAM_LANGUAGES.index("russian")
    assert codes.tolist() == [english, russian, OOV_ID, OOV_ID, NUM_LANGUAGE_IDS - 1]
    # a review scraped twice keeps one copy; unknown review ids are OOV
    batch = pa.table(
        {"rec_id": [7, 3, 7, None], "language": ["english", "russian", "english", "x"]}
    )
    lookup = ReviewLanguages.from_batches([batch])
    assert lookup.review_id.tolist() == [3, 7]
    assert lookup.of(np.array([7, 3, 5])).tolist() == [english, russian, OOV_ID]


def test_previous_language_is_strictly_before_the_row():
    # user 1: english at t=10, russian at t=20; user 2: schinese at t=5
    user = np.array([1, 2, 1])
    seconds = np.array([10, 5, 20])
    code = np.array([2, 3, 25], np.int8)
    rows_user = np.array([1, 1, 1, 2, 2, 9])
    rows_ts = np.array([10, 15, 21, 5, 6, 100])
    got = previous_language(user, seconds, code, rows_user, rows_ts)
    # t=10: its own review is not "previous"; unknown user 9 has none
    assert got.tolist() == [OOV_ID, 2, 25, OOV_ID, 3, OOV_ID]


def _raw_languages(marts) -> ReviewLanguages:
    """Each synthetic user writes in one language (by user id; some unknown)."""
    interactions = marts["interactions"]
    users = interactions["user_id"].to_numpy()
    names = [LANGS[u % len(LANGS)] for u in users]
    return ReviewLanguages.from_batches(
        [pa.table({"rec_id": interactions["review_id"], "language": pa.array(names, pa.string())})]
    )


@pytest.fixture(scope="module")
def loaded():
    marts = make_marts(n_users=200, reviews_per_user=10)
    source = FakeSource(marts)

    def load(**kwargs):
        return load_training_data(
            source,
            validation_fraction=0.1,
            cold_row_fraction=0.3,
            eval_max_rows=10**9,
            seed=0,
            with_tags=True,
            **kwargs,
        )

    return load(), load(languages=_raw_languages(marts)), marts


def test_loaded_rows_get_their_users_previous_language(loaded):
    plain, data, marts = loaded
    assert plain.train.language is None and plain.validation.language is None
    assert np.array_equal(data.train.target, plain.train.target)
    assert np.array_equal(data.validation.target, plain.validation.target)
    table = marts["interactions"]
    ts = table["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    users = table["user_id"].to_numpy()
    for i in range(len(data.validation)):
        user, t = data.validation.user_id[i], data.validation.timestamp[i]
        earlier = (users == user) & (ts < t)
        name = LANGS[user % len(LANGS)]
        want = (
            OOV_ID if not earlier.any() or name is None else FIRST_ID + STEAM_LANGUAGES.index(name)
        )
        assert data.validation.language[i] == want
    assert (data.train.language != OOV_ID).any()


def test_batches_carry_the_language_with_dropout(loaded):
    _, data, _ = loaded
    examples = build_examples(data.train, data.catalog, data.negatives)
    collator = Collator(
        examples, data.catalog, target_log_probs(examples.target, data.vocab.games), seed=0
    )
    idx = list(range(len(examples)))
    language = collator(idx).language.numpy()
    known = examples.language != OOV_ID
    kept = language[known] == examples.language[known]
    assert 0.8 < kept.mean() < 0.97  # LANGUAGE_DROPOUT
    assert LANGUAGE_DROPOUT == 0.1
    assert (language[known][~kept] == OOV_ID).all()
    plain = build_examples(loaded[0].train, data.catalog, data.negatives)
    no_language = Collator(plain, data.catalog, target_log_probs(plain.target, data.vocab.games))
    assert no_language(idx[:8]).language is None


def _model(user_language: bool) -> TwoTowerModel:
    vocab = VocabSizes(games=50, developers=10, publishers=10, genres=10, categories=10)
    config = ModelConfig(
        vocab=vocab,
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=16,
        output_dim=8,
        temperature=0.05,
        user_language=user_language,
    )
    torch.manual_seed(0)
    return TwoTowerModel(config).eval()


def test_user_tower_reads_the_language():
    model = _model(True)
    history = torch.tensor([[2, 3, 0, 0, 0]] * 3)
    language = torch.tensor([FIRST_ID, FIRST_ID + 1, OOV_ID])
    users = model.user_tower(history, language)
    assert not torch.allclose(users[0], users[1])
    # no language = unknown for every row
    assert torch.allclose(model.user_tower(history)[2], users[2])
    assert _model(False).user_tower(history, language).shape == (3, 8)  # ignored


def test_evaluation_scores_rows_with_their_language(loaded):
    _, data, _ = loaded
    rows = data.validation
    model = _model(True)
    with torch.no_grad():
        model.user_tower.language.weight.mul_(50)  # make the language matter
    score = evaluation.model_scores(model, data.catalog, 64)
    with_language, _ = evaluation.rank_rows(score, rows, data.catalog, 10, 64)
    from dataclasses import replace

    without, _ = evaluation.rank_rows(score, replace(rows, language=None), data.catalog, 10, 64)
    assert not torch.equal(with_language, without)


def test_settings_need_the_raw_reviews():
    with pytest.raises(ValidationError, match="RAW_REVIEWS_URI"):
        TrainingSettings(model_artifacts_bucket=BUCKET, model_id="m", user_language=True)


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_train_with_language_is_not_promotable(settings, source, store, marts, monkeypatch):
    monkeypatch.setattr(
        "steam_training.languages.load_review_languages", lambda *a, **k: _raw_languages(marts)
    )
    settings = settings.model_copy(
        update={
            "model_id": "a7",
            "epochs": 2,
            "user_language": True,
            "raw_reviews_uri": "s3://raw/reviews/",
        }
    )
    metadata = run_training(settings, source, store)
    assert metadata.config.user_language
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, _ = store.load_model("a7")
    assert model.user_tower.language is not None
    assert not store._exists(model_prefix("a7") + USER_TOWER_NUMPY)
    with pytest.raises(ValueError, match="not promotable"):
        run_promotion(settings, source, store)
