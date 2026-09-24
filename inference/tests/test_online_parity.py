"""Training's numpy user tower + this pipeline's online catalog, run by serving's numpy code,
reproduce the torch model exactly: same user embeddings (OOV, unseen games, padding, empty
histories) and the same top K as batch retrieval."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_model
from steam_serving.online import BundleManifest, OnlineModel, UserTower
from steam_training.export import encode_user_tower

from steam_inference.features import load_inference_data
from steam_inference.online import LocalBundleStore, build_catalog, encode_catalog
from steam_inference.retrieval import retrieve


@pytest.fixture
def setup(source, tmp_path):
    model = make_model(seed=4)
    seen = torch.ones(model.config.vocab.games, dtype=torch.bool)
    seen[[5, 9]] = False  # games without training interactions map to OOV
    model.set_seen_games(seen)
    data = load_inference_data(source)
    candidates = retrieve(
        model, data.games, data.users, data.reviews, k=5, user_batch_size=2, item_batch_size=7
    )
    store = LocalBundleStore(str(tmp_path / "online"))
    manifest_path = store.publish(
        encode_catalog(build_catalog(data.games, candidates.item_embeddings)),
        model_id="abc123",
        generated_at=datetime(2024, 7, 1, tzinfo=UTC),
        catalog_games=len(data.games),
        user_tower_key="models/abc123/user_tower.npz",
    )
    manifest = BundleManifest.model_validate_json(Path(manifest_path).read_text())
    catalog = (tmp_path / "online" / "catalog.npz").read_bytes()
    tower = UserTower.from_bytes(encode_user_tower(model, "abc123"))
    return model, data, candidates, OnlineModel.from_bytes(manifest, catalog, tower)


HISTORIES = [
    [7, 6, 5, 3, 2],  # 5 is unseen -> OOV
    [2, 0, 0, 0, 0],
    [21, 9, 4, 0, 0],  # 21 beyond the vocabulary, 9 unseen
    [0, 0, 0, 0, 0],  # empty
]


@pytest.mark.parametrize("history", HISTORIES)
def test_user_embedding_matches_torch(setup, history):
    model, _, _, online = setup
    with torch.no_grad():
        expected = model.user_tower(torch.tensor([history])).numpy()[0]
    np.testing.assert_allclose(online.user_tower.embed(np.array(history)), expected, atol=1e-6)


def test_names_and_ids(setup):
    _, data, _, online = setup
    assert online.catalog_games == len(data.games)
    assert [online.name(r) for r in range(len(data.games))] == list(data.games.name)


def test_same_top_k_as_batch_retrieval(setup):
    """Sending a user's history (most recent first) followed by the rest of their reviewed
    games gives exactly their batch candidates: same history, same exclusions."""
    _, data, candidates, online = setup
    u = data.users.user_id.tolist().index(102)  # 6 positive reviews: history = the last 5
    history_rows = data.games.catalog.rows(data.users.history[u][data.users.history[u] != 0])
    liked_ids = [int(data.games.game_id[r]) for r in history_rows]
    reviewed = data.reviews.of(data.users.take(np.array([u]))).row(0)
    older = [
        int(data.games.game_id[r])
        for r in data.games.catalog.rows(np.array(reviewed))
        if r >= 0 and int(data.games.game_id[r]) not in liked_ids
    ]
    result = online.recommend(liked_ids + older, k=5)
    assert result.used_game_ids == liked_ids
    batch_rows, batch_scores = candidates.of(u)
    np.testing.assert_allclose(result.scores, batch_scores, atol=1e-4)
    # exact ties (e.g. two OOV games with the same features) are ordered by lowest appid online;
    # torch.topk leaves their order unspecified
    batch = sorted(
        zip(batch_scores.round(4).tolist(), data.games.game_id[batch_rows].tolist(), strict=True),
        key=lambda pair: (-pair[0], pair[1]),
    )
    assert result.game_ids == [game_id for _, game_id in batch]
