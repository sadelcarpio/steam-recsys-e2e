"""Two-tower retrieval model.

Item tower: game_idx embedding + mean-pooled developer / publisher / genre / category embeddings
            + numerical features (is_free, is_free known, reviews ratio) -> MLP -> L2 norm.
User tower: mean-pooled `games_reviewed_positive` over a *frozen copy* of the item tower's game
            embedding table (+ history length) -> MLP -> L2 norm.

The copy (`UserTower.game_table`) is re-synced from the item tower once per epoch
(`TwoTowerModel.sync_user_table`): the user tower chases a target that only moves between
epochs, instead of both towers moving the same table inside every step (unstable gradients).

Ids outside a vocabulary (newer than the model) and games without training interactions map to
OOV, so new games are represented by the OOV row plus their content features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from steam_training.contracts import OOV_ID, PADDING_ID, USER_HISTORY_LENGTH, ModelConfig

NUM_NUMERIC_FEATURES = 3
ATTRIBUTES = ("developers", "publishers", "genres", "categories")


@dataclass
class Bag:
    """EmbeddingBag input: flat ids + start offset of each row."""

    values: torch.Tensor  # int64 [n_values]
    offsets: torch.Tensor  # int64 [rows]


@dataclass
class ItemBatch:
    game_idx: torch.Tensor  # int64 [rows]
    numeric: torch.Tensor  # float32 [rows, NUM_NUMERIC_FEATURES]
    developers: Bag
    publishers: Bag
    genres: Bag
    categories: Bag

    def __len__(self) -> int:
        return len(self.game_idx)

    def to(self, device: torch.device) -> ItemBatch:
        def bag(b: Bag) -> Bag:
            return Bag(b.values.to(device), b.offsets.to(device))

        return ItemBatch(
            game_idx=self.game_idx.to(device),
            numeric=self.numeric.to(device),
            developers=bag(self.developers),
            publishers=bag(self.publishers),
            genres=bag(self.genres),
            categories=bag(self.categories),
        )


def _clip_to_vocab(ids: torch.Tensor, size: int) -> torch.Tensor:
    return torch.where(ids < size, ids, torch.full_like(ids, OOV_ID))


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class _GameIdMapper(nn.Module):
    """Maps game_idx to itself when trained, OOV otherwise (padding stays padding)."""

    def __init__(self, num_games: int) -> None:
        super().__init__()
        self.num_games = num_games
        self.register_buffer("seen_games", torch.ones(num_games, dtype=torch.bool))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = _clip_to_vocab(ids, self.num_games)
        keep = self.seen_games[ids] | (ids == PADDING_ID)
        return torch.where(keep, ids, torch.full_like(ids, OOV_ID))


class ItemTower(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        vocab = config.vocab
        self.game_ids = _GameIdMapper(vocab.games)
        self.game_embedding = nn.Embedding(
            vocab.games, config.game_embedding_dim, padding_idx=PADDING_ID
        )
        self.attribute_sizes = {name: getattr(vocab, name) for name in ATTRIBUTES}
        self.attributes = nn.ModuleDict(
            {
                name: nn.EmbeddingBag(
                    size, config.attribute_embedding_dim, mode="mean", padding_idx=PADDING_ID
                )
                for name, size in self.attribute_sizes.items()
            }
        )
        in_dim = (
            config.game_embedding_dim
            + len(ATTRIBUTES) * config.attribute_embedding_dim
            + NUM_NUMERIC_FEATURES
        )
        self.mlp = _mlp(in_dim, config.hidden_dim, config.output_dim)

    def forward(self, items: ItemBatch) -> torch.Tensor:
        parts = [self.game_embedding(self.game_ids(items.game_idx))]
        for name in ATTRIBUTES:
            bag: Bag = getattr(items, name)
            values = _clip_to_vocab(bag.values, self.attribute_sizes[name])
            parts.append(self.attributes[name](values, bag.offsets))
        parts.append(items.numeric)
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=-1)


class UserTower(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        vocab = config.vocab
        self.game_ids = _GameIdMapper(vocab.games)
        # Frozen snapshot of ItemTower.game_embedding, refreshed once per epoch.
        self.register_buffer("game_table", torch.zeros(vocab.games, config.game_embedding_dim))
        self.mlp = _mlp(config.game_embedding_dim + 1, config.hidden_dim, config.output_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: int64 [rows, USER_HISTORY_LENGTH], most recent first, 0 = padding."""
        ids = self.game_ids(history)
        present = (ids != PADDING_ID).unsqueeze(-1).float()
        lengths = present.sum(dim=1)
        pooled = (self.game_table[ids] * present).sum(dim=1) / lengths.clamp(min=1)
        features = torch.cat([pooled, lengths / USER_HISTORY_LENGTH], dim=1)
        return F.normalize(self.mlp(features), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)

    @torch.no_grad()
    def sync_user_table(self) -> None:
        self.user_tower.game_table.copy_(self.item_tower.game_embedding.weight.detach())

    @torch.no_grad()
    def set_seen_games(self, seen: torch.Tensor) -> None:
        seen = seen.to(torch.bool).clone()
        seen[PADDING_ID] = True
        seen[OOV_ID] = True
        self.user_tower.game_ids.seen_games.copy_(seen)
        self.item_tower.game_ids.seen_games.copy_(seen)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Similarity logits [users, items] (cosine / temperature)."""
        return users @ items.T / self.config.temperature
