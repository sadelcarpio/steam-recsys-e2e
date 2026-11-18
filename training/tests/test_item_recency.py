"""A6 (spec 17): item recency features (reviews in the 4 / 13 weeks before, game age) from
`game_popularity_weekly`, as of the review's ISO week in training and as of a weekly refreshed
catalog in evaluation."""

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
    POPULARITY_MART,
    Recency,
    WeeklyReviews,
    load_training_data,
    release_days,
    week_of_day,
    week_start_day,
)
from steam_training.model import NUM_NUMERIC_FEATURES, TwoTowerModel
from steam_training.model import NUM_RECENCY_FEATURES as MODEL_RECENCY_FEATURES
from steam_training.pipeline import run_promotion, run_training
from steam_training.ranker.features import week_of
from tests.conftest import BUCKET, FakeSource, make_marts


def test_weeks_are_the_marts_and_the_rankers_iso_weeks():
    seconds = np.random.default_rng(0).integers(0, 2_000_000_000, 1000)
    day = seconds // DAY_SECONDS
    assert np.array_equal(week_of_day(day), week_of(seconds))
    monday = week_start_day(week_of_day(day))
    assert ((monday + 3) % 7 == 0).all()  # day 4 (1970-01-05) was a Monday
    assert ((day - monday >= 0) & (day - monday < 7)).all()


def test_weekly_reviews_count_the_weeks_strictly_before():
    rng = np.random.default_rng(0)
    game = rng.integers(2, 6, 500)
    week = rng.integers(2800, 2830, 500)
    reviews = rng.integers(1, 5, 500)
    counts = WeeklyReviews.from_rows(game, week, reviews)
    query_game = rng.integers(0, 8, 200)
    query_week = rng.integers(2795, 2835, 200)
    for weeks in (1, 4, 13):
        got = counts.window(query_game, query_week, weeks)
        want = [
            int(reviews[(game == g) & (week < w) & (week >= w - weeks)].sum())
            for g, w in zip(query_game, query_week, strict=True)
        ]
        assert got.tolist() == want


def test_release_days_parse_exact_dates_only():
    details = pa.table(
        {
            "game_id": pa.array([10, 11, 12, 13, 14, 99], pa.int64()),
            "game_release_date": pa.array(
                ["Jul 9, 2021", "Coming soon", "2027", None, " 14 Nov, 2025", "Jan 1, 2020"]
            ),
        }
    )
    lookup = pa.table({"game_idx": [2, 3, 4, 5, 6], "game_id": [10, 11, 12, 13, 14]})
    days = release_days(details.to_batches(), lookup.to_batches(), num_games=8)
    epoch_day = lambda y, m, d: (datetime(y, m, d) - datetime(1970, 1, 1)).days  # noqa: E731
    # appid 99 has no game_idx; padding / OOV / unknown dates stay -1
    assert days.tolist() == [-1, -1, epoch_day(2021, 7, 9), -1, -1, -1, epoch_day(2025, 11, 14), -1]


def test_recency_features_are_as_of_the_iso_week():
    day = 20_000  # a Saturday: its week starts on Monday, day 19_996
    w = int(week_of_day(day))
    assert week_start_day(w) == 19_996
    counts = WeeklyReviews.from_rows(
        np.array([2, 2, 2, 3, 2, 2]),
        np.array([w - 1, w - 5, w - 13, w - 1, w - 14, w]),  # w itself: not counted yet
        np.array([1, 2, 3, 4, 5, 6]),
    )
    release = np.array([-1, -1, 19_900, -1], np.int32)
    recency = Recency(counts, release)
    out = recency.features(np.array([2, 2, 3]), np.array([day, 19_996, day]))
    assert out.shape == (3, NUM_RECENCY_FEATURES) == (3, MODEL_RECENCY_FEATURES)
    expected_2 = [0.1 * np.log1p(1), 0.1 * np.log1p(1 + 2 + 3), 0.1 * np.log1p(96), 1]
    np.testing.assert_allclose(
        out,
        [
            # 4 weeks = {w-1}, 13 weeks = {w-1, w-5, w-13}, age at Monday = 19_996 - 19_900
            expected_2,
            expected_2,  # the Monday of the same week: the same features
            [0.1 * np.log1p(4), 0.1 * np.log1p(4), 0.0, 0],  # release unknown
        ],
        rtol=1e-6,
    )
    # the next week counts week w, drops w-13 from the 13 weeks
    nxt = recency.features(np.array([2]), np.array([day + 7]))[0]
    np.testing.assert_allclose(nxt[:2], [0.1 * np.log1p(6 + 1), 0.1 * np.log1p(6 + 1 + 2)])


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


def _reviews_per_game_week(source) -> tuple[np.ndarray, np.ndarray]:
    table = source.tables["interactions"]
    seconds = table["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    return table["game_idx"].to_numpy(), week_of_day(seconds // DAY_SECONDS)


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
    assert "game_details" in recency.snapshots and POPULARITY_MART in recency.snapshots
    # the catalog's 4-week counts: every review of the game (the mart counts interactions) in
    # the 4 ISO weeks before the cutoff's week
    game, week = _reviews_per_game_week(source)
    cutoff_week = int(week_of_day(cutoff_day))
    items = recency.catalog.items
    want = [
        ((game == g) & (week < cutoff_week) & (week >= cutoff_week - 4)).sum()
        for g in items.game_idx
    ]
    assert sum(want) > 0
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
    start = int(week_of_day(data.catalog.as_of_day))
    row_week = np.maximum(week_of_day(rows.timestamp // DAY_SECONDS), start)
    weeks = [int(week_start_day(w)) for w in sorted(set(row_week.tolist()) - {start})]
    assert days == weeks and len(weeks) > 2
    # the same ranks as scoring each week's rows against that week's catalog by hand
    for as_of in weeks[:2]:
        part = np.flatnonzero(row_week == week_of_day(as_of))
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


def test_train_and_promote_with_recency(settings, source, store):
    settings = settings.model_copy(update={"model_id": "a6", "epochs": 2, "item_recency": True})
    metadata = run_training(settings, source, store)
    assert metadata.config.item_recency
    assert metadata.validation.warm.recall[5] > metadata.popularity_baseline.warm.recall[5]
    model, _ = store.load_model("a6")
    assert model.item_tower.numeric_dim == NUM_NUMERIC_FEATURES + NUM_RECENCY_FEATURES
    record = store.load_tracking("a6")
    assert record is not None and record.recent_popularity is not None
    # promotion scores each row against the catalog as of its week, like training's evaluation
    report = run_promotion(settings, source, store)
    assert report.promoted
    assert report.candidate.metrics.warm.recall[5] == pytest.approx(
        metadata.validation.warm.recall[5], abs=0.05
    )


def test_weekly_scores_refresh_only_recency_models(loaded):
    _, data, _ = loaded
    catalog = data.catalog
    assert catalog.as_of_day is not None
    start = int(week_of_day(catalog.as_of_day))
    history = torch.from_numpy(data.validation.history[:4])
    for item_recency in (False, True):
        model = _model(item_recency=item_recency)
        scores_at = evaluation.weekly_scores(model, catalog, batch_size=16)
        assert scores_at(start - 2) is scores_at(start)  # earlier weeks keep the catalog's
        later = scores_at(start + 20)
        assert (later is scores_at(start)) == (not item_recency)
        if item_recency:
            assert later is scores_at(start + 20)  # cached
            assert not torch.allclose(later(history), scores_at(start)(history))


def test_weekly_top_candidates_score_each_row_as_of_its_week(loaded):
    _, data, _ = loaded
    rows = data.validation.take(np.arange(12))
    index = evaluation.CatalogIndex.of(data.catalog)
    calls = []

    def scores_at(week):
        calls.append(week)

        # week w prefers catalog row w % len: each row's top game tells its week
        def score(history, language=None):
            out = torch.zeros(len(history), len(index))
            out[:, week % len(index)] = 1.0
            return out

        return score

    seconds = (week_start_day(np.arange(3000, 3012) // 4 * 4) * 86_400).astype(np.int64)
    games, _ = evaluation.weekly_top_candidates(
        scores_at, np.zeros_like(rows.history), seconds, index, 3, batch_size=5
    )
    weeks = week_of_day(seconds // 86_400)
    assert sorted(calls) == sorted(set(weeks.tolist()))
    expected = index.game_idx[torch.from_numpy(weeks % len(index))].numpy()
    assert np.array_equal(games[:, 0], expected)


def test_time_buckets_are_iso_weeks_with_sparse_old_weeks_merged():
    monday = 19_996  # week w starts here
    # 3 rows in week w, 1 in w + 1, 5 in w + 2, 2 in w + 3 (Saturday / Sunday / Monday edges)
    day = monday + np.array([0, 4, 6, 7, 14, 15, 16, 19, 20, 21, 27])
    bucket, starts = time_buckets(day, min_rows=4)
    # weeks w and w + 1 merge (3 + 1 rows), w + 2 has 5, w + 3 keeps its 2 (the newest)
    assert starts.tolist() == [monday, monday + 14, monday + 21]
    assert bucket.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert np.array_equal(starts, week_start_day(week_of_day(starts)))  # all Mondays


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
    bucket, starts = time_buckets(examples.day, min_rows=64)
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
