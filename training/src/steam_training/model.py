"""Two-tower retrieval model.

Item tower: game_idx embedding + mean-pooled developer / publisher / genre / category embeddings
            + weighted mean of the game's Steam user tag embeddings (when `vocab.tags` is set)
            + numerical features (is_free, is_free known, reviews ratio) -> MLP -> L2 norm.
User tower: mean-pooled `games_reviewed_positive` over a *frozen copy* of the item tower's game
            embedding table (+ history length) [+ tag profile of the history, A4] -> MLP ->
            L2 norm. The tag profile is a weighted bag of the history games' Steam tags
            (tag weight x a learned weight per history position, normalized per user) over
            the user tower's own tag embeddings. It reads the raw game ids, so games without
            training interactions (OOV in the game table) still contribute their tags.

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
    """EmbeddingBag input: flat ids + start offset of each row (+ per-value weights)."""

    values: torch.Tensor  # int64 [n_values]
    offsets: torch.Tensor  # int64 [rows]
    weights: torch.Tensor | None = None  # float32 [n_values]

    def to(self, device: torch.device) -> Bag:
        weights = self.weights.to(device) if self.weights is not None else None
        return Bag(self.values.to(device), self.offsets.to(device), weights)


@dataclass
class ItemBatch:
    game_idx: torch.Tensor  # int64 [rows]
    numeric: torch.Tensor  # float32 [rows, NUM_NUMERIC_FEATURES]
    developers: Bag
    publishers: Bag
    genres: Bag
    categories: Bag
    tags: Bag  # weighted (each row sums to 1); ignored by a model without tags

    def __len__(self) -> int:
        return len(self.game_idx)

    def to(self, device: torch.device) -> ItemBatch:
        return ItemBatch(
            game_idx=self.game_idx.to(device),
            numeric=self.numeric.to(device),
            developers=self.developers.to(device),
            publishers=self.publishers.to(device),
            genres=self.genres.to(device),
            categories=self.categories.to(device),
            tags=self.tags.to(device),
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
        # Steam user tags: a weighted mean (weights sum to 1 per game, so mode "sum"). Only in
        # models trained with tags, so older models keep their state dict layout.
        self.tag_size = vocab.tags
        self.tags = (
            nn.EmbeddingBag(
                vocab.tags, config.attribute_embedding_dim, mode="sum", padding_idx=PADDING_ID
            )
            if vocab.tags is not None
            else None
        )
        in_dim = (
            config.game_embedding_dim
            + (len(ATTRIBUTES) + (self.tags is not None)) * config.attribute_embedding_dim
            + NUM_NUMERIC_FEATURES
        )
        self.mlp = _mlp(in_dim, config.hidden_dim, config.output_dim)

    def forward(self, items: ItemBatch) -> torch.Tensor:
        parts = [self.game_embedding(self.game_ids(items.game_idx))]
        for name in ATTRIBUTES:
            bag: Bag = getattr(items, name)
            values = _clip_to_vocab(bag.values, self.attribute_sizes[name])
            parts.append(self.attributes[name](values, bag.offsets))
        if self.tags is not None:
            bag = items.tags
            values = _clip_to_vocab(bag.values, self.tag_size)
            parts.append(self.tags(values, bag.offsets, per_sample_weights=bag.weights))
        parts.append(items.numeric)
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=-1)


_TAG_BUFFERS = ("game_tag_values", "game_tag_offsets", "game_tag_weights")


class UserTower(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        vocab = config.vocab
        self.num_games = vocab.games
        self.game_ids = _GameIdMapper(vocab.games)
        # Frozen snapshot of ItemTower.game_embedding, refreshed once per epoch.
        self.register_buffer("game_table", torch.zeros(vocab.games, config.game_embedding_dim))
        in_dim = config.game_embedding_dim + 1
        self.tag_profile: nn.EmbeddingBag | None = None
        if config.user_tag_profile:
            if vocab.tags is None:
                raise ValueError("user_tag_profile needs a tags vocabulary")
            self.tag_size = vocab.tags
            # game_idx -> tags (CSR, set from the training catalog; sized on load)
            self.register_buffer("game_tag_values", torch.zeros(0, dtype=torch.int64))
            self.register_buffer(
                "game_tag_offsets", torch.zeros(vocab.games + 1, dtype=torch.int64)
            )
            self.register_buffer("game_tag_weights", torch.zeros(0))
            # learned weight of each history position (exp, so positive; starts equal)
            self.tag_recency = nn.Parameter(torch.zeros(USER_HISTORY_LENGTH))
            self.tag_profile = nn.EmbeddingBag(
                vocab.tags, config.attribute_embedding_dim, mode="sum", padding_idx=PADDING_ID
            )
            in_dim += config.attribute_embedding_dim
        self.mlp = _mlp(in_dim, config.hidden_dim, config.output_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # the tag CSR's length depends on the catalog: take it from the saved tensors
        if self.tag_profile is not None:
            for name in _TAG_BUFFERS:
                saved = state_dict.get(prefix + name)
                if saved is not None:
                    current = getattr(self, name)
                    setattr(self, name, torch.empty_like(saved, device=current.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def set_game_tags(self, values: torch.Tensor, offsets: torch.Tensor, weights) -> None:
        device = self.game_tag_offsets.device
        self.game_tag_values = values.to(device=device, dtype=torch.int64)
        self.game_tag_offsets = offsets.to(device=device, dtype=torch.int64)
        self.game_tag_weights = weights.to(device=device, dtype=torch.float32)

    def _tag_profile(self, history: torch.Tensor) -> torch.Tensor:
        """Weighted bag of the history games' tags, normalized per user (no tags: zeros)."""
        assert self.tag_profile is not None
        rows, slots = history.shape
        games = _clip_to_vocab(history, self.num_games).reshape(-1)  # padding: no tags
        starts = self.game_tag_offsets[games]
        lengths = self.game_tag_offsets[games + 1] - starts
        slot = torch.arange(rows * slots, device=history.device)
        owner_slot = torch.repeat_interleave(slot, lengths)
        first = torch.cumsum(lengths, 0) - lengths
        within = torch.arange(len(owner_slot), device=history.device) - first[owner_slot]
        source = starts[owner_slot] + within
        position_weight = torch.exp(self.tag_recency)[owner_slot % slots]
        weights = self.game_tag_weights[source] * position_weight
        user = owner_slot // slots
        totals = torch.zeros(rows, device=history.device).index_add(0, user, weights)
        weights = weights / totals[user].clamp(min=1e-12)
        per_user = torch.zeros(rows, dtype=torch.int64, device=history.device).index_add(
            0, user, torch.ones_like(user)
        )
        offsets = torch.cumsum(per_user, 0) - per_user
        values = _clip_to_vocab(self.game_tag_values[source], self.tag_size)
        return self.tag_profile(values, offsets, per_sample_weights=weights)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: int64 [rows, USER_HISTORY_LENGTH], most recent first, 0 = padding."""
        ids = self.game_ids(history)
        present = (ids != PADDING_ID).unsqueeze(-1).float()
        lengths = present.sum(dim=1)
        pooled = (self.game_table[ids] * present).sum(dim=1) / lengths.clamp(min=1)
        parts = [pooled, lengths / USER_HISTORY_LENGTH]
        if self.tag_profile is not None:
            parts.append(self._tag_profile(history))
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=-1)


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

    def set_game_tags(self, tags) -> None:
        """A4: the game -> tags lookup of the user tower's tag profile (a `data.Ragged` by
        game_idx, e.g. `Catalog.game_tags`)."""
        self.user_tower.set_game_tags(
            torch.from_numpy(tags.values),
            torch.from_numpy(tags.offsets),
            torch.from_numpy(tags.weights),
        )

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Similarity logits [users, items] (cosine / temperature)."""
        return users @ items.T / self.config.temperature
