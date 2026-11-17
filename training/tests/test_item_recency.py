"""A6 (spec 17): item recency features (reviews in the last 30 / 90 days, game age) as of the
review's day in training and as of a weekly refreshed catalog in evaluation."""

from datetime import datetime

import boto3
import numpy as np
import pyarrow as pa
import pytest
import torch
from moto import mock_aws

from steam_training import evaluation
from steam_training.artifacts import ArtifactStore
from steam_training.batching import (
    BucketBatchSampler,
    Collator,
    bucket_log_probs,
    build_examples,
    target_log_probs,
    time_buckets,
)
from steam_training.contracts import ModelConfig, VocabSizes
from steam_training.data import (
    DAY_SECONDS,
    NUM_RECENCY_FEATURES,
    Recency,
    ReviewCounts,
    load_training_data,
    release_days,
)
from steam_training.model import NUM_NUMERIC_FEATURES, TwoTowerModel
from steam_training.model import NUM_RECENCY_FEATURES as MODEL_RECENCY_FEATURES
from steam_training.pipeline import run_promotion, run_training
from tests.conftest import BUCKET, FakeSource, make_marts


def _counts(game, day) -> ReviewCounts:
    game, day = np.asarray(game), np.asarray(day)
    return ReviewCounts.from_parts(
        [np.unique(ReviewCounts.key(game[:3], day[:3]), return_counts=True)]
        + [np.unique(ReviewCounts.key(game[3:], day[3:]), return_counts=True)]
    )


def test_review_counts_are_strictly_before_the_day_within_the_window():
    rng = np.random.default_rng(0)
    game = rng.integers(2, 6, 500)
    day = rng.integers(20_000, 20_200, 500)
    counts = _counts(game, day)
    query_game = rng.integers(0, 8, 200)
    query_day = rng.integers(19_990, 20_220, 200)
    for days in (1, 30, 90):
        got = counts.window(query_game, query_day, days)
        want = [
            int(((game == g) & (day < d) & (day >= d - days)).sum())
            for g, d in zip(query_game, query_day, strict=True)
        ]
        assert got.tolist() == want


def test_release_days_parse_exact_dates_only():
    details = pa.table(
        {
            "game_id": pa.array([10, 11, 12, 13, 14, 99], pa.int64()),
            "game_release_date": pa.array(
                ["Jul 9, 2021", "Coming soon", "2027", None, "Nov 14, 2025", "Jan 1, 2020"]
            ),
        }
    )
    lookup = pa.table({"game_idx": [2, 3, 4, 5, 6], "game_id": [10, 11, 12, 13, 14]})
    days = release_days(details.to_batches(), lookup.to_batches(), num_games=8)
    epoch_day = lambda y, m, d: (datetime(y, m, d) - datetime(1970, 1, 1)).days  # noqa: E731
    # appid 99 has no game_idx; padding / OOV / unknown dates stay -1
    assert days.tolist() == [-1, -1, epoch_day(2021, 7, 9), -1, -1, -1, epoch_day(2025, 11, 14), -1]


def test_recency_features():
    counts = _counts([2, 2, 2, 3, 2], [100, 150, 199, 199, 50])
    release = np.array([-1, -1, 120, -1], np.int32)
    out = Recency(counts, release).features(np.array([2, 2, 3]), np.array([200, 110, 200]))
    assert out.shape == (3, NUM_RECENCY_FEATURES) == (3, MODEL_RECENCY_FEATURES)
    np.testing.assert_allclose(
        out,
        [
            # day 200: 30 days = {199}, 90 days = {150, 199}, released on day 120
            [0.1 * np.log1p(1), 0.1 * np.log1p(2), 0.1 * np.log1p(80), 1],
            # day 110: 30 days = {100}, 90 days = {50, 100}, before release: age 0
            [0.1 * np.log1p(1), 0.1 * np.log1p(2), 0.0, 1],
            [0.1 * np.log1p(1), 0.1 * np.log1p(1), 0.0, 0],  # release unknown
        ],
        rtol=1e-6,
    )


def _load(source, **kwargs):
    return load_training_data(
        source,
        validation_fraction=0.1,
        cold_row_fraction=0.3,
        eval_max_rows=10**9,
        seed=0,
        with_tags=True,
        **kwargs,
    )


@pytest.fixture(scope="module")
def loaded():
    source = FakeSource(make_marts(n_users=200, reviews_per_user=14))
    return _load(source), _load(source, with_recency=True), source


def _reviews_per_game_day(source) -> tuple[np.ndarray, np.ndarray]:
    table = source.tables["interactions"]
    seconds = table["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    return table["game_idx"].to_numpy(), seconds // DAY_SECONDS


def test_loading_with_recency_keeps_the_rows_and_dates_the_catalog(loaded):
    plain, recency, source = loaded
    assert plain.catalog.recency is None and plain.train.day is None
    assert plain.catalog.items.recency is None
    # the same rows, now with their day
    assert np.array_equal(recency.train.target, plain.train.target)
    assert np.array_equal(recency.validation.target, plain.validation.target)
    cutoff_day = int(np.datetime64(recency.split.cutoff, "s").astype(np.int64)) // DAY_SECONDS
    assert recency.catalog.as_of_day == cutoff_day
    assert (recency.train.day <= cutoff_day).all()
    assert "game_details" in recency.snapshots
    # the catalog's 30-day counts: every review of the game in the 30 days before the cutoff day
    game, day = _reviews_per_game_day(source)
    items = recency.catalog.items
    want = [
        ((game == g) & (day < cutoff_day) & (day >= cutoff_day - 30)).sum() for g in items.game_idx
    ]
    np.testing.assert_allclose(items.recency[:, 0], 0.1 * np.log1p(want), rtol=1e-6)
    # exact release dates are known (synthetic: game_idx % 10 in 7, 8, 9 are not)
    assert items.recency[:, 3].tolist() == [float(g % 10 < 7) for g in items.game_idx]


def test_batches_date_targets_and_hard_negatives_by_their_review(loaded):
    _, data, _ = loaded
    examples = build_examples(data.train, data.catalog, data.negatives)
    collator = Collator(
        examples, data.catalog, target_log_probs(examples.target, data.vocab.games), seed=1
    )
    idx = list(range(0, len(examples), 7))
    batch = collator(idx)
    numeric = batch.items.numeric.numpy()
    assert numeric.shape[1] == NUM_NUMERIC_FEATURES + NUM_RECENCY_FEATURES
    day = examples.day[idx]
    want = data.catalog.recency.features(examples.target[idx], day)
    np.testing.assert_allclose(numeric[:, NUM_NUMERIC_FEATURES:], want, rtol=1e-6)
    explicit = batch.negatives[0]
    valid = explicit.valid.numpy()
    assert valid.any()
    games = explicit.game_idx.numpy()[valid]
    got = explicit.items.numeric.numpy()[valid, NUM_NUMERIC_FEATURES:]
    np.testing.assert_allclose(got, data.catalog.recency.features(games, day[valid]), rtol=1e-6)


def _model(item_recency: bool) -> TwoTowerModel:
    vocab = VocabSizes(games=50, developers=10, publishers=10, genres=10, categories=10, tags=10)
    config = ModelConfig(
        vocab=vocab,
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=16,
        output_dim=8,
        temperature=0.05,
        item_recency=item_recency,
    )
    torch.manual_seed(0)
    return TwoTowerModel(config).eval()


def test_models_without_recency_ignore_the_extra_columns(loaded):
    plain, recency, _ = loaded
    model = _model(item_recency=False)
    a = evaluation.embed_catalog(model, plain.catalog, batch_size=16)
    b = evaluation.embed_catalog(model, recency.catalog, batch_size=16)
    assert torch.allclose(a, b)
    with pytest.raises(ValueError, match="A6"):
        evaluation.embed_catalog(_model(item_recency=True), plain.catalog, batch_size=16)


def test_evaluation_refreshes_the_catalog_weekly_for_recency_models(loaded):
    _, data, _ = loaded
    rows = data.validation
    model = _model(item_recency=True)
    base = evaluation.model_scores(model, data.catalog, 64)
    days = []

    def rescore(catalog):
        days.append(catalog.as_of_day)
        return evaluation.model_scores(model, catalog, 64)

    refreshed, _ = evaluation.rank_rows(base, rows, data.catalog, 10, 64, rescore=rescore)
    start = data.catalog.as_of_day
    row_days = rows.timestamp // DAY_SECONDS
    weeks = sorted({start + (int(d) - start) // 7 * 7 for d in row_days} - {start})
    assert days == weeks and len(weeks) > 2
    # the same ranks as scoring each week's rows against that week's catalog by hand
    week = start + (row_days - start) // 7 * 7
    for as_of in weeks[:2]:
        part = np.flatnonzero(week == as_of)
        catalog = data.catalog.at_day(as_of)
        by_hand, _ = evaluation.rank_rows(
            evaluation.model_scores(model, catalog, 64), rows.take(part), catalog, 10, 64
        )
        assert torch.equal(refreshed[torch.from_numpy(part)], by_hand)
    # and the catalog really moves between weeks
    later = data.catalog.at_day(weeks[-1]).items.recency
    assert not np.array_equal(later[:, :2], data.catalog.items.recency[:, :2])


@pytest.fixture
def store(aws_env):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield ArtifactStore(BUCKET, s3)


def test_train_with_recency_is_not_promotable(settings, source, store):
    settings = settings.model_copy(update={"model_id": "a6", "epochs": 2, "item_recency": True})
    metadata = run_training(settings, source, store)
    assert metadata.config.item_recency
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, _ = store.load_model("a6")
    assert model.item_tower.numeric_dim == NUM_NUMERIC_FEATURES + NUM_RECENCY_FEATURES
    record = store.load_tracking("a6")
    assert record is not None and record.recent_popularity is not None
    with pytest.raises(ValueError, match="not promotable"):
        run_promotion(settings, source, store)


def test_time_buckets_are_the_evaluation_weeks_with_sparse_old_weeks_merged():
    anchor = 1000
    # weeks before the anchor: 3 rows in week -3, 1 in week -2, 5 in week -1, 2 in week 0
    day = np.array([980, 980, 981, 986, 993, 993, 994, 995, 999, 1000, 1006])
    bucket, starts = time_buckets(day, anchor, min_rows=4)
    # weeks -3 and -2 merge (3 + 1 rows), week -1 has 5, week 0 keeps its 2 (the newest)
    assert starts.tolist() == [979, 993, 1000]
    assert bucket.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    # each bucket starts on an evaluation week boundary (weeks anchored at the catalog's day)
    assert ((starts - anchor) % 7 == 0).all()


def test_bucket_log_probs_are_target_frequencies_within_the_bucket():
    target = np.array([5, 5, 6, 7, 5, 5])
    bucket = np.array([0, 0, 0, 0, 1, 1])
    np.testing.assert_allclose(
        np.exp(bucket_log_probs(target, bucket)), [0.5, 0.5, 0.25, 0.25, 1.0, 1.0], rtol=1e-6
    )


def test_bucket_batches_cover_every_example_once_and_never_mix_buckets():
    bucket = np.repeat(np.arange(5), [100, 37, 5, 260, 61])
    rng = np.random.default_rng(0)
    bucket = bucket[rng.permutation(len(bucket))]
    sampler = BucketBatchSampler(bucket, batch_size=50, generator=torch.Generator().manual_seed(3))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 2 + 1 + 1 + 5 + 1
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(len(bucket)))
    assert all(len(set(bucket[b].tolist())) == 1 for b in batches)
    # near-equal splits: no tiny batch takes a full step
    assert min(len(b) for b in batches if len(b) != 5) >= 37
    again = list(BucketBatchSampler(bucket, 50, torch.Generator().manual_seed(3)))
    assert [list(b) for b in again] == [list(b) for b in batches]


def test_collator_dates_a_batch_by_its_bucket_and_uses_the_bucket_log_q(loaded):
    _, data, _ = loaded
    examples = build_examples(data.train, data.catalog, data.negatives)
    bucket, starts = time_buckets(examples.day, data.catalog.as_of_day, min_rows=64)
    examples.as_of = starts[bucket].astype(np.int32)
    examples.log_q = bucket_log_probs(examples.target, bucket)
    collator = Collator(
        examples, data.catalog, target_log_probs(examples.target, data.vocab.games), seed=1
    )
    idx = np.flatnonzero(bucket == bucket[0])[:32].tolist()
    batch = collator(idx)
    as_of = np.full(len(idx), starts[bucket[0]])
    want = data.catalog.recency.features(examples.target[idx], as_of)
    np.testing.assert_allclose(
        batch.items.numeric.numpy()[:, NUM_NUMERIC_FEATURES:], want, rtol=1e-6
    )
    np.testing.assert_allclose(batch.target_log_prob.numpy(), examples.log_q[idx])


def test_recent_popularity_ranks_this_weeks_most_reviewed_games(loaded):
    plain, data, _ = loaded
    score = evaluation.recent_popularity_scores(data.catalog)
    scores = score(torch.zeros(2, 5, dtype=torch.int64))
    recency = data.catalog.items.recency
    best = int(np.argmax(recency[:, 0] + 1e-3 * recency[:, 1]))
    assert scores.shape == (2, len(data.catalog)) and int(scores[0].argmax()) == best
    with pytest.raises(ValueError, match="A6"):
        evaluation.recent_popularity_scores(plain.catalog)
    # refreshed weekly like an A6 model
    ranks, _ = evaluation.rank_rows(
        score, data.validation, data.catalog, 10, 64, rescore=evaluation.recent_popularity_rescore()
    )
    assert len(ranks) == len(data.validation)
