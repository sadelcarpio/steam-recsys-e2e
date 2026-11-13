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
            A3: `history_length` games (5 = the mart's history); with `history_encoder =
            "attention"` a self-attention summary of the history (`HistoryAttention`) is
            appended to the mean (which stays, so attention only adds to the baseline).

The copy (`UserTower.game_table`) is re-synced from the item tower once per epoch
(`TwoTowerModel.sync_user_table`): the user tower chases a target that only moves between
epochs, instead of both towers moving the same table inside every step (unstable gradients).

Ids outside a vocabulary (newer than the model) and games without training interactions map to
OOV, so new games are represented by the OOV row plus their content features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from steam_training.contracts import OOV_ID, PADDING_ID, ModelConfig

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


class _AttentionBlock(nn.Module):
    """Pre-LN transformer block: x + MHA(LN(x)), then x + FFN(LN(x)) (ReLU, width 2 x dim).
    Written out (not nn.TransformerEncoderLayer) so a numpy port can mirror it exactly."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = _mlp(dim, 2 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """x: [rows, tokens, dim]; keep: bool [rows, tokens], the keys that can be attended."""
        rows, tokens, dim = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(rows, tokens, self.heads, dim // self.heads).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dim // self.heads)
        scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = (weights @ v).transpose(1, 2).reshape(rows, tokens, dim)
        x = x + self.dropout(self.out(mixed))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class HistoryAttention(nn.Module):
    """A3: self-attention over [summary token, history games + learned position embeddings]
    (position 0 = the most recent game). Returns the summary token's final state (LayerNorm),
    a learned attention pooling of the history. The summary token is always attendable, so an
    empty history is well defined (it attends to itself only)."""

    def __init__(self, dim: int, length: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.summary = nn.Parameter(torch.randn(dim) * 0.02)
        self.position = nn.Parameter(torch.randn(length, dim) * 0.02)
        self.blocks = nn.ModuleList(_AttentionBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, games: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """games: [rows, length, dim] (padding rows are zeros); present: bool [rows, length]."""
        rows = games.shape[0]
        x = torch.cat([self.summary.expand(rows, 1, -1), games + self.position], dim=1)
        keep = torch.cat([present.new_ones(rows, 1), present], dim=1)
        for block in self.blocks:
            x = block(x, keep)
        return self.norm(x[:, 0])


class UserTower(nn.Module):
    def __init__(self, config: ModelConfig, attention_dropout: float = 0.0) -> None:
        super().__init__()
        vocab = config.vocab
        self.num_games = vocab.games
        self.history_length = config.history_length
        self.game_ids = _GameIdMapper(vocab.games)
        # Frozen snapshot of ItemTower.game_embedding, refreshed once per epoch.
        self.register_buffer("game_table", torch.zeros(vocab.games, config.game_embedding_dim))
        in_dim = config.game_embedding_dim + 1
        self.attention: HistoryAttention | None = None
        if config.history_encoder == "attention":
            self.attention = HistoryAttention(
                config.game_embedding_dim,
                config.history_length,
                config.attention_heads,
                config.attention_layers,
                attention_dropout,
            )
            in_dim += config.game_embedding_dim
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
            self.tag_recency = nn.Parameter(torch.zeros(config.history_length))
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
        """history: int64 [rows, history_length], most recent first, 0 = padding."""
        if history.shape[1] != self.history_length:
            raise ValueError(
                f"history has {history.shape[1]} games, the model reads {self.history_length}"
            )
        ids = self.game_ids(history)
        present = ids != PADDING_ID
        weights = present.unsqueeze(-1).float()
        lengths = weights.sum(dim=1)
        games = self.game_table[ids]
        pooled = (games * weights).sum(dim=1) / lengths.clamp(min=1)
        parts = [pooled, lengths / self.history_length]
        if self.attention is not None:
            parts.append(self.attention(games * weights, present))
        if self.tag_profile is not None:
            parts.append(self._tag_profile(history))
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, config: ModelConfig, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.config = config
        self.user_tower = UserTower(config, attention_dropout)
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
