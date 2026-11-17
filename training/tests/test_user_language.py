"""A7 (spec 17): the language of the user's previous review as a user tower input, read from the
`review_languages` mart."""

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
from steam_training.contracts import (
    FIRST_ID,
    NUM_LANGUAGE_IDS,
    OOV_ID,
    STEAM_LANGUAGES,
    ModelConfig,
    VocabSizes,
)
from steam_training.data import load_training_data, previous_language
from steam_training.export import USER_TOWER_LANGUAGE_FORMAT, user_tower_arrays
from steam_training.languages import (
    LANGUAGE_MART,
    ReviewLanguages,
    language_codes,
    load_review_languages,
)
from steam_training.model import TwoTowerModel
from steam_training.pipeline import run_promotion, run_training
from steam_training.tracking import features
from tests.conftest import BUCKET, FakeSource, make_marts

LANGS = ("english", "schinese", "russian", None)


def test_language_codes_and_lookup():
    codes = language_codes(pa.array(["english", "Russian", "klingon", None, "ukrainian"]))
    english, russian = FIRST_ID, FIRST_ID + STEAM_LANGUAGES.index("russian")
    assert codes.tolist() == [english, russian, OOV_ID, OOV_ID, NUM_LANGUAGE_IDS - 1]
    # a review scraped twice keeps one copy; unknown review ids are OOV
    batch = pa.table(
        {"review_id": [7, 3, 7, None], "language": ["english", "russian", "english", "x"]}
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


def with_languages(marts) -> dict:
    """+ the `review_languages` mart: each synthetic user writes in one language (by user id);
    reviews without a language are absent, as in the mart."""
    interactions = marts["interactions"]
    users = interactions["user_id"].to_numpy()
    names = [LANGS[u % len(LANGS)] for u in users]
    keep = [n is not None for n in names]
    table = pa.table(
        {"review_id": interactions["review_id"], "language": pa.array(names, pa.string())}
    ).filter(pa.array(keep))
    return {**marts, LANGUAGE_MART: table}


def _raw_languages(marts) -> ReviewLanguages:
    return load_review_languages(FakeSource(with_languages(marts)))


def test_load_review_languages_keeps_only_the_asked_reviews():
    marts = with_languages(make_marts(n_users=20, reviews_per_user=4))
    everything = load_review_languages(FakeSource(marts))
    assert len(everything.review_id) == marts[LANGUAGE_MART].num_rows
    wanted = np.array([0, 1, 2, 5, 10**9])
    some = load_review_languages(FakeSource(marts), review_ids=wanted)
    assert set(some.review_id.tolist()) <= set(wanted.tolist())
    np.testing.assert_array_equal(some.of(wanted), everything.of(wanted))


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


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_train_and_promote_with_language(settings, store, marts):
    source = FakeSource(with_languages(marts))
    settings = settings.model_copy(update={"model_id": "a7", "epochs": 2, "user_language": True})
    metadata = run_training(settings, source, store)
    assert metadata.config.user_language
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, _ = store.load_model("a7")
    assert model.user_tower.language is not None
    assert store._exists(model_prefix("a7") + USER_TOWER_NUMPY)  # format 4
    report = run_promotion(settings, source, store)
    assert report.promoted
    assert report.candidate.metrics.warm.recall[5] == pytest.approx(
        metadata.validation.warm.recall[5], abs=0.05
    )


def test_language_export_is_format_4():
    model = _model(True)
    arrays = user_tower_arrays(model, "a7")
    assert int(arrays["format_version"]) == USER_TOWER_LANGUAGE_FORMAT
    assert arrays["language_table"].shape == (NUM_LANGUAGE_IDS, 8)
    np.testing.assert_array_equal(
        arrays["language_table"], model.user_tower.language.weight.detach().numpy()
    )
    assert tuple(arrays["language_names"].tolist()) == STEAM_LANGUAGES
    assert int(user_tower_arrays(_model(False), "m")["format_version"]) == 1


def test_models_saved_with_a_retired_field_still_load():
    saved = _model(False).config
    config = saved.model_dump()
    assert ModelConfig.model_validate({**config, "history_gap_seconds": None}) == saved
    with pytest.raises(ValidationError, match="retired"):
        ModelConfig.model_validate({**config, "history_gap_seconds": 3600})
    assert "previous_review_language" in features(_model(True).config)["user"]
