import numpy as np
import torch

from steam_training.data import Catalog, EvalRows, ItemFeatures, Ragged
from steam_training.evaluation import (
    NOT_FOUND,
    CatalogIndex,
    evaluate,
    mask_games,
    mine_hard_negatives,
    popularity_scores,
    recall_from_ranks,
    target_ranks,
)


def catalog(games: list[int]) -> Catalog:
    n = len(games)
    items = ItemFeatures(
        game_idx=np.array(games, dtype=np.int64),
        is_free=np.zeros(n, np.float32),
        is_free_known=np.ones(n, np.float32),
        reviews_ratio=np.full(n, 0.5, np.float32),
        developers=Ragged.from_lists([[2]] * n),
        publishers=Ragged.from_lists([[2]] * n),
        genres=Ragged.from_lists([[2]] * n),
        categories=Ragged.from_lists([[2]] * n),
    )
    row_of = np.full(max(games) + 1, -1, dtype=np.int64)
    row_of[games] = np.arange(n)
    return Catalog(items=items, row_of=row_of)


CATALOG = catalog([2, 3, 4, 5, 6])
INDEX = CatalogIndex.of(CATALOG)
# fixed ranking 6 > 5 > 4 > 3 > 2 for everybody
FIXED = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


def fixed_scores(history: torch.Tensor) -> torch.Tensor:
    return FIXED.repeat(len(history), 1)


def test_ranks_exclude_history_items():
    history = torch.tensor([[0] * 5, [6, 5, 0, 0, 0], [0] * 5])
    target = torch.tensor([4, 4, 99])
    ranks = target_ranks(fixed_scores, history, target, INDEX, max_k=5, batch_size=2)
    # row 0: 6,5,4 -> rank 2; row 1: 6,5 masked -> 4 first; row 2: unknown target
    assert ranks.tolist() == [2, 0, NOT_FOUND]


def test_recall_segments():
    ranks = torch.tensor([0, 3, NOT_FOUND, 1])
    warm = torch.tensor([True, True, False, False])
    metrics = recall_from_ranks(ranks, warm, [1, 2, 5])
    assert metrics.warm.recall == {1: 0.5, 2: 0.5, 5: 1.0}
    assert metrics.cold.recall == {1: 0.0, 2: 0.5, 5: 0.5}
    assert metrics.all.rows == 4 and metrics.all.recall[2] == 0.5


def test_empty_segment_is_zero():
    metrics = recall_from_ranks(torch.tensor([0]), torch.tensor([True]), [1])
    assert metrics.cold.rows == 0 and metrics.cold.recall == {1: 0.0}


def test_popularity_baseline_ranks_by_training_counts():
    score = popularity_scores(np.bincount([3, 3, 3, 5, 5, 2]), CATALOG)
    rows = EvalRows(np.zeros((2, 5), np.int64), np.array([3, 5]), np.array([True, True]))
    metrics = evaluate(score, rows, CATALOG, [1, 2], batch_size=8)
    assert metrics.warm.recall == {1: 0.5, 2: 1.0}


def test_mask_games_sets_minus_inf_and_ignores_unknown_games():
    history = torch.tensor([[3, 99, 0, 0, -1]])
    scores = mask_games(fixed_scores(history), history, INDEX)
    assert torch.isinf(scores[0, 1]) and torch.isfinite(scores[0, [0, 2, 3, 4]]).all()


def test_catalog_index_rows():
    assert INDEX.rows(torch.tensor([2, 6, 0, 99, -1])).tolist() == [0, 4, -1, -1, -1]


def test_mining_samples_from_the_requested_rank_window(monkeypatch):
    import steam_training.evaluation as evaluation

    monkeypatch.setattr(evaluation, "model_scores", lambda *a, **k: fixed_scores)
    history = np.array([[5, 0, 0, 0, 0]] * 200)
    target = np.array([6] * 200)
    mined = mine_hard_negatives(
        None,
        history,
        target,
        CATALOG,
        skip_top=1,
        pool_size=2,
        generator=torch.Generator().manual_seed(0),
        batch_size=64,
    )
    # 6 (target) and 5 (history) are excluded: ranking 4, 3, 2 -> window [1, 3) = {3, 2}
    assert set(mined.tolist()) == {3, 2}
