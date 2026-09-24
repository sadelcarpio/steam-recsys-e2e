import numpy as np
import torch
from conftest import REVIEWS, make_model
from steam_training.batching import to_item_batch

from steam_inference.features import load_inference_data
from steam_inference.retrieval import NO_ROW, retrieve


def _retrieve(source, k=5):
    data = load_inference_data(source)
    model = make_model()
    candidates = retrieve(
        model, data.games, data.users, data.reviews, k=k, user_batch_size=3, item_batch_size=4
    )
    return model, data, candidates


def test_matches_brute_force_without_reviewed_games(source):
    model, data, candidates = _retrieve(source)
    with torch.no_grad():
        catalog = data.games.catalog.items
        items = model.item_tower(to_item_batch(catalog.take(np.arange(len(catalog)))))
        scores = (model.user_tower(torch.from_numpy(data.users.history)) @ items.T).numpy()
    for u, user in enumerate(data.users.user_id):
        reviewed = {g for g, _ in REVIEWS[int(user)]}
        allowed = [
            r
            for r in range(len(data.games))
            if data.games.catalog.items.game_idx[r] not in reviewed
        ]
        expected = sorted(allowed, key=lambda r: -scores[u, r])[:5]
        rows, top_scores = candidates.of(u)
        assert rows.tolist() == expected
        assert np.allclose(top_scores, scores[u, expected], atol=1e-5)
        assert not reviewed & set(data.games.catalog.items.game_idx[rows].tolist())


def test_fewer_candidates_than_k(source):
    _, data, candidates = _retrieve(source)
    u104 = data.users.user_id.tolist().index(104)
    assert (candidates.rows[u104] != NO_ROW).sum() == 2  # reviewed all but two games
    assert len(candidates.of(u104)[0]) == 2


def test_k_larger_than_catalog(source):
    _, data, candidates = _retrieve(source, k=1000)
    assert candidates.rows.shape == (len(data.users), len(data.games))


def test_new_games_are_scored_from_content(source):
    """The last game is newer than the model's vocabulary: it maps to OOV, still ranked."""
    model, data, _ = _retrieve(source)
    new_game = data.games.catalog.items.game_idx.max()
    assert new_game >= model.config.vocab.games
    candidates = retrieve(
        model, data.games, data.users, data.reviews, k=1000, user_batch_size=8, item_batch_size=8
    )
    u102 = data.users.user_id.tolist().index(102)
    ranked = data.games.catalog.items.game_idx[candidates.of(u102)[0]]
    assert new_game in ranked
