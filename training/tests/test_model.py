import numpy as np
import torch

from steam_training.batching import to_item_batch
from steam_training.contracts import OOV_ID, ModelConfig, VocabSizes
from steam_training.data import ItemFeatures, Ragged
from steam_training.model import TwoTowerModel

VOCAB = VocabSizes(games=10, developers=5, publishers=5, genres=4, categories=4)


def config() -> ModelConfig:
    return ModelConfig(
        vocab=VOCAB,
        game_embedding_dim=8,
        attribute_embedding_dim=4,
        hidden_dim=16,
        output_dim=6,
        temperature=0.1,
    )


def items(game_idx: list[int], developers: list[list[int]] | None = None) -> ItemFeatures:
    n = len(game_idx)
    developers = developers or [[2]] * n
    return ItemFeatures(
        game_idx=np.array(game_idx, dtype=np.int64),
        is_free=np.zeros(n, np.float32),
        is_free_known=np.ones(n, np.float32),
        reviews_ratio=np.full(n, 0.5, np.float32),
        developers=Ragged.from_lists(developers),
        publishers=Ragged.from_lists([[2]] * n),
        genres=Ragged.from_lists([[2, 3]] * n),
        categories=Ragged.from_lists([[]] * n),
    )


def test_towers_output_unit_vectors():
    model = TwoTowerModel(config())
    model.sync_user_table()
    item = model.item_tower(to_item_batch(items([2, 3, 4], [[2], [], [3, 4]])))
    user = model.user_tower(torch.tensor([[2, 3, 0, 0, 0], [0, 0, 0, 0, 0]]))
    assert item.shape == (3, 6) and user.shape == (2, 6)
    assert torch.allclose(item.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(user.norm(dim=1), torch.ones(2), atol=1e-5)


def test_ids_beyond_vocab_and_unseen_games_map_to_oov():
    model = TwoTowerModel(config())
    seen = torch.zeros(VOCAB.games, dtype=torch.bool)
    seen[2] = True
    model.set_seen_games(seen)
    mapper = model.item_tower.game_ids
    assert mapper(torch.tensor([0, 1, 2, 3, 99])).tolist() == [0, OOV_ID, 2, OOV_ID, OOV_ID]
    # attribute ids beyond the vocabulary do not crash (mapped to OOV)
    model.item_tower(to_item_batch(items([2], [[999]])))
    # an unseen game has the same embedding as OOV (content features equal)
    out = model.item_tower(to_item_batch(items([3, OOV_ID])))
    assert torch.allclose(out[0], out[1])


def test_user_table_is_a_frozen_copy_synced_per_epoch():
    model = TwoTowerModel(config())
    assert not model.user_tower.game_table.requires_grad
    model.sync_user_table()
    assert torch.equal(model.user_tower.game_table, model.item_tower.game_embedding.weight)
    # user tower gradients never reach the item tower's game embedding
    loss = model.user_tower(torch.tensor([[2, 3, 4, 0, 0]])).sum()
    loss.backward()
    assert model.item_tower.game_embedding.weight.grad is None
    with torch.no_grad():
        model.item_tower.game_embedding.weight.add_(1.0)
    assert not torch.equal(model.user_tower.game_table, model.item_tower.game_embedding.weight)
    model.sync_user_table()
    assert torch.equal(model.user_tower.game_table, model.item_tower.game_embedding.weight)


def test_padding_positions_do_not_change_the_user_embedding():
    model = TwoTowerModel(config())
    model.sync_user_table()
    with torch.no_grad():
        model.user_tower.game_table[0] = 100.0  # padding row must be ignored
    a = model.user_tower(torch.tensor([[2, 3, 0, 0, 0]]))
    model.sync_user_table()
    b = model.user_tower(torch.tensor([[2, 3, 0, 0, 0]]))
    assert torch.allclose(a, b)
