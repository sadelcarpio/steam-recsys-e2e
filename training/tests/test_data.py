from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from steam_training.data import (
    GAME_FEATURE_COLUMNS,
    EvalRows,
    Ragged,
    catalog_at,
    load_training_data,
    temporal_cutoff,
)
from tests.conftest import FIRST_GAME, N_GAMES, FakeSource


def load(source, **kwargs):
    options = {
        "validation_fraction": 0.1,
        "cold_row_fraction": 0.0,
        "eval_max_rows": 10**9,
        "seed": 0,
    }
    return load_training_data(source, **{**options, **kwargs})


def test_ragged_from_arrow_treats_null_lists_as_empty():
    column = pa.chunked_array([pa.array([[2, 3], None]), pa.array([[], [4]])])
    ragged = Ragged.from_arrow(column)
    assert [ragged.row(i) for i in range(len(ragged))] == [[2, 3], [], [], [4]]


def test_ragged_take_gathers_rows_in_order():
    ragged = Ragged.from_lists([[1], [2, 3], [], [4, 5, 6]])
    taken = ragged.take(np.array([3, 0, 2, 1, 3]))
    assert [taken.row(i) for i in range(len(taken))] == [[4, 5, 6], [1], [], [2, 3], [4, 5, 6]]
    assert taken.offsets.tolist() == [0, 3, 4, 4, 6, 9]


def test_temporal_cutoff_leaves_about_the_fraction_for_validation():
    ts = np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(100)])
    ts = ts.astype("datetime64[us]")
    cutoff = temporal_cutoff(np.random.default_rng(0).permutation(ts), 0.1)
    assert cutoff == datetime(2024, 3, 31)  # day 90 (leap year)
    assert (ts >= np.datetime64(cutoff)).sum() == 10


def test_temporal_cutoff_rejects_empty():
    with pytest.raises(ValueError):
        temporal_cutoff(np.array([], dtype="datetime64[us]"), 0.1)


def test_load_keeps_exactly_the_rows_training_needs(marts, source):
    data = load(source)
    table = marts["interactions"]
    ts = table["timestamp"].to_numpy()
    is_val = ts >= np.datetime64(data.split.cutoff)
    positive = table["is_positive"].to_numpy(zero_copy_only=False)
    warm = np.array([h[0] != 0 for h in table["games_reviewed_positive"].to_pylist()])
    game = table["game_idx"].to_numpy()

    assert data.vocab.games == FIRST_GAME + N_GAMES and data.vocab.genres == FIRST_GAME + 4
    assert set(data.snapshots) >= {"interactions", "game_features", "lkp_games"}
    assert data.split.train_rows + data.split.validation_rows == table.num_rows
    assert 0.08 < data.split.validation_rows / table.num_rows < 0.12
    # cold_row_fraction=0: warm training positives only
    assert len(data.train) == (positive & warm & ~is_val).sum()
    assert (data.train.history[:, 0] != 0).all()
    assert len(data.negatives) == (~positive & ~is_val).sum()
    assert len(data.validation) == (positive & is_val).sum()
    assert (
        data.train_positive_counts.tolist()
        == np.bincount(game[positive & ~is_val], minlength=data.vocab.games).tolist()
    )
    # as-of numerical features come from the row
    assert np.allclose(data.train.reviews_ratio, 0.9)


def test_load_samples_cold_rows_and_caps_validation(source):
    full = load(source, cold_row_fraction=1.0)
    half = load(source, cold_row_fraction=0.5)
    cold_full = (full.train.history[:, 0] == 0).sum()
    cold_half = (half.train.history[:, 0] == 0).sum()
    assert 0.3 * cold_full < cold_half < 0.7 * cold_full
    capped = load(source, eval_max_rows=50)
    assert 25 < len(capped.validation) < 80  # Bernoulli sample around the cap


def test_load_is_independent_of_batch_size(marts):
    a = load(FakeSource(marts, batch_rows=13), cold_row_fraction=0.3)
    b = load(FakeSource(marts, batch_rows=100_000), cold_row_fraction=0.3)
    assert a.split == b.split
    assert np.array_equal(a.validation.target, b.validation.target)
    assert np.array_equal(a.catalog.items.game_idx, b.catalog.items.game_idx)
    # sampling draws per batch, so only the deterministic parts match exactly
    assert (a.train.history[:, 0] != 0).sum() == (b.train.history[:, 0] != 0).sum()


def test_load_for_evaluation_only_at_a_given_cutoff(source):
    data = load(source, cutoff=datetime(2024, 7, 1), with_train_rows=False)
    assert data.train is None and data.negatives is None
    assert data.split.cutoff == datetime(2024, 7, 1)
    assert data.split.validation_rows / (data.split.train_rows + data.split.validation_rows) > 0.4


def test_catalog_takes_latest_row_before_cutoff(marts):
    batches = lambda: marts["game_features"].select(GAME_FEATURE_COLUMNS).to_batches(7)  # noqa: E731
    early = catalog_at(batches(), datetime(2024, 3, 1), FIRST_GAME + N_GAMES)
    late = catalog_at(batches(), datetime(2025, 1, 1), FIRST_GAME + N_GAMES)
    assert len(early) == len(late) == N_GAMES
    assert np.allclose(early.items.reviews_ratio, 0.5)
    assert np.allclose(late.items.reviews_ratio, 0.7)
    rows = late.rows(np.array([FIRST_GAME, 0, 10_000, -1]))
    assert late.items.game_idx[rows[0]] == FIRST_GAME
    assert rows[1:].tolist() == [-1, -1, -1]
    with pytest.raises(ValueError):
        catalog_at(batches(), datetime(1960, 1, 1), FIRST_GAME + N_GAMES)


def test_null_labels_and_features_are_handled(marts):
    table = marts["interactions"]
    nulled = table.set_column(
        table.schema.get_field_index("is_positive"),
        "is_positive",
        pc.if_else(pc.equal(table["review_id"], 0), None, table["is_positive"]),
    ).set_column(
        table.schema.get_field_index("game_is_free"),
        "game_is_free",
        pa.nulls(table.num_rows, pa.bool_()),
    )
    data = load(FakeSource({**marts, "interactions": nulled}))
    assert (data.train.is_free_known == 0).all() and (data.train.is_free == 0).all()


def test_eval_rows_sample():
    rows = EvalRows(np.zeros((10, 5), np.int64), np.arange(10), np.ones(10, bool))
    assert len(rows.sample(3)) == 3 and rows.sample(20) is rows
