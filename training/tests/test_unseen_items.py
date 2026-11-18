"""Unseen items: the exposure check over the current catalog (`evaluation.concentration`, the
pair gate) and the id dropout that trains the OOV row on hard negatives and new releases."""

import boto3
import numpy as np
import pytest
import torch
from moto import mock_aws
from pydantic import ValidationError

from steam_training import evaluation
from steam_training.artifacts import ArtifactStore
from steam_training.batching import Collator, build_examples, target_log_probs
from steam_training.config import PairSettings, TrainingSettings
from steam_training.contracts import OOV_ID, Concentration, PairEvaluation, RankingMetrics
from steam_training.data import current_catalog, load_training_data, week_of_day
from steam_training.pairs import decide_pair
from steam_training.pipeline import run_training
from steam_training.tracking import final_metrics
from tests.conftest import BUCKET, N_GAMES, FakeSource, make_marts


@pytest.fixture(scope="module")
def marts():
    return make_marts(n_users=200, reviews_per_user=14)


@pytest.fixture(scope="module")
def data(marts):
    return load_training_data(
        FakeSource(marts),
        validation_fraction=0.1,
        cold_row_fraction=0.3,
        eval_max_rows=10**9,
        seed=0,
        with_tags=True,
        with_recency=True,
    )


# ---- the exposure measure ----


def test_concentration_counts_reach_top10_share_and_unseen_slots(data):
    index = evaluation.CatalogIndex.of(data.catalog)
    rows = len(index)
    favourite = int(index.game_idx[3])

    def score(history, language=None):
        # every user: the favourite first, then a per-user rotation
        out = torch.rand(len(history), rows, generator=torch.Generator().manual_seed(0))
        out[:, 3] = 10.0
        return out

    seen = np.ones(data.vocab.games, bool)
    seen[favourite] = False  # the favourite is a game the model never trained on
    history = np.zeros((50, 5), np.int64)
    c = evaluation.concentration(score, history, index, seen, k=4, batch_size=16)
    assert (c.k, c.lists, c.top_game_idx) == (4, 50, favourite)
    assert c.reach == 1.0
    assert c.unseen_share == pytest.approx(0.25)  # 1 of every 4 slots
    assert 0.25 < c.top10_share <= 1.0 and c.distinct > 4
    # the history is excluded: a user who reviewed the favourite never gets it
    history[0, 0] = favourite
    c = evaluation.concentration(score, history, index, seen, k=4, batch_size=16)
    assert c.reach == pytest.approx(49 / 50)


def test_current_catalog_has_every_game_with_recency_as_of_the_newest_week(marts, data):
    source = FakeSource(marts)
    catalog = current_catalog(source, data.vocab.games, with_tags=True, with_recency=True)
    # unreleased games ("Coming soon": game_idx % 10 == 7) are never recommended: left out
    assert len(catalog) == N_GAMES - 4
    assert not (catalog.items.game_idx % 10 == 7).any()
    # newest features: the conftest's second game_features row (ratio 0.7)
    assert np.allclose(catalog.items.reviews_ratio, 0.7)
    weeks = marts["game_popularity_weekly"]["week_start"].to_numpy()
    newest = int(week_of_day(np.datetime64(max(weeks), "D").astype(np.int64)))
    assert week_of_day(catalog.as_of_day) == newest
    assert catalog.items.recency is not None and catalog.items.tags is not None
    plain = current_catalog(source, data.vocab.games, with_tags=False, with_recency=False)
    assert plain.items.recency is None and plain.as_of_day is None


# ---- the pair gate ----


def _metrics(r: float) -> RankingMetrics:
    values = {20: r, 50: r, 100: r}
    return RankingMetrics(groups=10, recall=values, ndcg=values)


def _exposure(reach: float | None) -> Concentration | None:
    if reach is None:
        return None
    return Concentration(
        k=10, lists=100, reach=reach, top_game_idx=7, top10_share=0.3, unseen_share=0.2,
        distinct=500,
    )  # fmt: skip


def _evaluation(
    retriever: str,
    recall: float,
    reach: float | None,
    final_reach: float | None = None,
    ranker: str | None = None,
) -> PairEvaluation:
    return PairEvaluation(
        retriever_id=retriever,
        ranker_id=ranker,
        final=_metrics(recall),
        retrieval=_metrics(recall),
        concentration=_exposure(reach),
        final_concentration=_exposure(final_reach if final_reach is not None else reach),
    )


def test_pair_gate_rejects_a_retriever_that_floods_one_game():
    settings = PairSettings(model_artifacts_bucket=BUCKET, retriever_id="new")
    champion = _evaluation("old", 0.10, 0.12)
    flooded = _evaluation("new", 0.20, 0.61)  # better recall, one game for 61% of users
    chosen, reason = decide_pair(champion, [flooded], True, settings)
    assert chosen is None and "PAIR_MAX_GAME_REACH" in reason and "61.0%" in reason
    fine = _evaluation("new", 0.20, 0.18)
    assert decide_pair(champion, [fine], True, settings)[0] is fine
    # reports written before the check have no concentration: not gated
    assert decide_pair(champion, [_evaluation("new", 0.2, None)], True, settings)[0] is not None
    looser = settings.model_copy(update={"pair_max_game_reach": 0.7})
    assert decide_pair(champion, [flooded], True, looser)[0] is flooded


def test_pair_gate_rejects_a_ranker_that_floods_one_game():
    """The ranker re-orders the retriever's top 100: its final lists are gated too, and a
    flooding pair is skipped for the next best one."""
    settings = PairSettings(model_artifacts_bucket=BUCKET, retriever_id="new")
    champion = _evaluation("old", 0.10, 0.12)
    with_ranker = _evaluation("new", 0.25, 0.18, final_reach=0.45, ranker="k")
    alone = _evaluation("new", 0.20, 0.18)
    chosen, _ = decide_pair(champion, [with_ranker, alone], True, settings)
    assert chosen is alone  # higher recall, but the ranker floods
    chosen, reason = decide_pair(champion, [with_ranker], True, settings)
    assert chosen is None and "every challenger floods" in reason and "45.0%" in reason


# ---- the id dropout ----


def _collator(data, **kwargs) -> tuple[Collator, object]:
    examples = build_examples(data.train, data.catalog, data.negatives)
    examples.mined = examples.target[::-1].copy()  # a hard negative per example
    log_probs = target_log_probs(examples.target, data.vocab.games)
    return Collator(examples, data.catalog, log_probs, seed=3, **kwargs), examples


def test_hard_negatives_keep_their_ids_by_default(data):
    collator, examples = _collator(data, item_id_dropout=0.0)
    batch = collator(list(range(64)))
    for negatives in batch.negatives:
        ids = negatives.items.game_idx[negatives.valid]
        assert (ids != OOV_ID).all()


def test_hard_negative_id_dropout_maps_them_to_oov(data):
    collator, _ = _collator(data, item_id_dropout=0.0, hard_negative_id_dropout=1.0)
    batch = collator(list(range(64)))
    assert (batch.items.game_idx != OOV_ID).all()  # targets untouched
    for negatives in batch.negatives:
        valid = negatives.valid
        assert valid.any()
        assert (negatives.items.game_idx[valid] == OOV_ID).all()
        # the mask in the loss still knows which game it is
        assert (negatives.game_idx[valid] != OOV_ID).all()


def test_new_item_id_dropout_drops_only_recent_releases(data):
    collator, examples = _collator(data, item_id_dropout=0.0, new_item_id_dropout=1.0)
    idx = list(range(len(examples)))[:256]
    batch = collator(idx)
    day = examples.as_of[idx] if examples.as_of is not None else examples.day[idx]
    release = data.catalog.recency.release_day[examples.target[idx]]
    age = day.astype(np.int64) - release
    young = (release >= 0) & (age >= 0) & (age <= 90)
    assert young.any() and (~young).any()
    dropped = batch.items.game_idx.numpy() == OOV_ID
    np.testing.assert_array_equal(dropped, young)


def test_new_item_id_dropout_needs_recency():
    with pytest.raises(ValidationError, match="NEW_ITEM_ID_DROPOUT"):
        TrainingSettings(model_artifacts_bucket=BUCKET, model_id="m", new_item_id_dropout=0.5)
    TrainingSettings(
        model_artifacts_bucket=BUCKET, model_id="m", new_item_id_dropout=0.5, item_recency=True
    )


# ---- training records it ----


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_training_records_the_concentration(settings, store, marts):
    source = FakeSource(marts)
    settings = settings.model_copy(
        update={
            "model_id": "c1",
            "epochs": 1,
            "item_recency": True,
            "hard_negative_id_dropout": 0.2,
            "new_item_id_dropout": 0.5,
        }
    )
    run_training(settings, source, store)
    record = store.load_tracking("c1")
    assert record is not None and record.concentration is not None
    c = record.concentration
    assert c.k == settings.concentration_k and c.lists > 0
    assert 0 < c.reach <= 1 and 0 <= c.unseen_share <= 1
    metrics = final_metrics(record)
    assert metrics["concentration/top_game_reach_at_10"] == c.reach
    assert "concentration/unseen_share_at_10" in metrics
    # off: no record
    run_training(
        settings.model_copy(update={"model_id": "c2", "concentration_rows": 0}), source, store
    )
    assert store.load_tracking("c2").concentration is None
