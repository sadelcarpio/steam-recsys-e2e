"""Stage 2, DataLoader / collate: training examples, hard negatives and batch assembly.

A training example is a kept positive interaction (`data.TrainRows`): the user's history as of
before the review is the query, the reviewed game is the target. Each example also carries:
  - an explicit hard negative: a game the same user reviewed negatively (training split);
  - a mined hard negative: refreshed at the start of each epoch (`evaluation.mine_hard_negatives`).
Other examples' targets in the batch are the in-batch negatives.

Target features: list features from the catalog (static per game), numerical features as of the
review. Collation is numpy; the batch becomes tensors at the end (`Batch.to(device)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from steam_training.contracts import OOV_ID, PADDING_ID
from steam_training.data import Catalog, ItemFeatures, NegativeRows, Ragged, TrainRows
from steam_training.model import Bag, ItemBatch

NO_GAME = -1


@dataclass
class TrainingExamples:
    history: np.ndarray  # int64 [n, 5]
    target: np.ndarray  # int64 [n] game_idx
    catalog_row: np.ndarray  # int64 [n] catalog row of the target
    # numerical target features as of the review
    is_free: np.ndarray
    is_free_known: np.ndarray
    reviews_ratio: np.ndarray
    # explicit negatives of the example's user: negative_games[start:start + count]
    negative_start: np.ndarray
    negative_count: np.ndarray
    negative_games: np.ndarray
    # mined hard negative per example (game_idx or NO_GAME), refreshed every epoch
    mined: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def build_examples(
    train: TrainRows, catalog: Catalog, negatives: NegativeRows | None = None
) -> TrainingExamples:
    n = len(train)
    start = np.zeros(n, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    negative_games = np.zeros(0, dtype=np.int64)
    if negatives is not None and len(negatives):
        order = np.argsort(negatives.user_id, kind="stable")
        neg_users = negatives.user_id[order]
        negative_games = negatives.game_idx[order]
        start = np.searchsorted(neg_users, train.user_id, side="left")
        count = np.searchsorted(neg_users, train.user_id, side="right") - start
    return TrainingExamples(
        history=train.history,
        target=train.target,
        catalog_row=catalog.rows(train.target),
        is_free=train.is_free,
        is_free_known=train.is_free_known,
        reviews_ratio=train.reviews_ratio,
        negative_start=start,
        negative_count=count,
        negative_games=negative_games,
        mined=np.full(n, NO_GAME, dtype=np.int64),
    )


def target_log_probs(target: np.ndarray, num_games: int) -> np.ndarray:
    """log P(game is a batch item) under uniform example sampling, for the logQ correction."""
    counts = np.bincount(target, minlength=num_games).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.log(counts / max(counts.sum(), 1.0)).astype(np.float32)


def to_bag(ragged: Ragged) -> Bag:
    return Bag(torch.from_numpy(ragged.values), torch.from_numpy(ragged.offsets[:-1].copy()))


def to_item_batch(
    items: ItemFeatures, game_idx: np.ndarray | None = None, numeric: np.ndarray | None = None
) -> ItemBatch:
    if numeric is None:
        numeric = np.stack([items.is_free, items.is_free_known, items.reviews_ratio], axis=1)
    return ItemBatch(
        game_idx=torch.from_numpy(items.game_idx if game_idx is None else game_idx),
        numeric=torch.from_numpy(numeric.astype(np.float32)),
        developers=to_bag(items.developers),
        publishers=to_bag(items.publishers),
        genres=to_bag(items.genres),
        categories=to_bag(items.categories),
    )


@dataclass
class HardNegatives:
    items: ItemBatch  # one row per example (placeholder features where invalid)
    game_idx: torch.Tensor  # int64 [B]
    valid: torch.Tensor  # bool [B]

    def to(self, device: torch.device) -> HardNegatives:
        return HardNegatives(self.items.to(device), self.game_idx.to(device), self.valid.to(device))


@dataclass
class Batch:
    history: torch.Tensor  # int64 [B, 5]
    items: ItemBatch  # targets
    target: torch.Tensor  # int64 [B] true game_idx (before id dropout)
    target_log_prob: torch.Tensor  # float32 [B]
    negatives: list[HardNegatives]

    def __len__(self) -> int:
        return len(self.target)

    def to(self, device: torch.device) -> Batch:
        return Batch(
            history=self.history.to(device),
            items=self.items.to(device),
            target=self.target.to(device),
            target_log_prob=self.target_log_prob.to(device),
            negatives=[n.to(device) for n in self.negatives],
        )


def history_dropout(history: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Truncate warm histories (>= 2 games) to a random shorter most-recent prefix w.p. p."""
    history = history.copy()
    lengths = (history != PADDING_ID).sum(axis=1)
    drop = (lengths >= 2) & (rng.random(len(history)) < p)
    if drop.any():
        new_lengths = np.ones(len(history), dtype=np.int64)
        new_lengths[drop] = rng.integers(1, lengths[drop])  # in [1, length - 1]
        cut = drop[:, None] & (np.arange(history.shape[1])[None, :] >= new_lengths[:, None])
        history[cut] = PADDING_ID
    return history


class Collator:
    """Builds a `Batch` from example indices (all numpy, vectorised)."""

    def __init__(
        self,
        examples: TrainingExamples,
        catalog: Catalog,
        log_probs: np.ndarray,
        *,
        history_dropout: float = 0.0,
        item_id_dropout: float = 0.0,
        explicit_negatives: bool = True,
        mined_negatives: bool = True,
        seed: int = 0,
    ) -> None:
        self.examples = examples
        self.catalog = catalog
        self.log_probs = log_probs
        self.history_dropout = history_dropout
        self.item_id_dropout = item_id_dropout
        self.explicit_negatives = explicit_negatives
        self.mined_negatives = mined_negatives
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, epoch: int) -> None:
        """Per-epoch stream, so a resumed run draws the same batches as an uninterrupted one."""
        self.rng = np.random.default_rng([self.seed, epoch])

    def __call__(self, indices: list[int]) -> Batch:
        idx = np.asarray(indices, dtype=np.int64)
        ex = self.examples
        target = ex.target[idx]
        history = history_dropout(ex.history[idx], self.history_dropout, self.rng)
        dropped = self.rng.random(len(idx)) < self.item_id_dropout
        numeric = np.stack([ex.is_free[idx], ex.is_free_known[idx], ex.reviews_ratio[idx]], axis=1)
        items = to_item_batch(
            self.catalog.items.take(ex.catalog_row[idx]),
            game_idx=np.where(dropped, OOV_ID, target),
            numeric=numeric,
        )
        negatives = []
        if self.explicit_negatives:
            negatives.append(self._hard_negatives(self._sample_explicit(idx), target))
        if self.mined_negatives:
            negatives.append(self._hard_negatives(ex.mined[idx], target))
        return Batch(
            history=torch.from_numpy(history),
            items=items,
            target=torch.from_numpy(target),
            target_log_prob=torch.from_numpy(self.log_probs[target]),
            negatives=negatives,
        )

    def _sample_explicit(self, idx: np.ndarray) -> np.ndarray:
        ex = self.examples
        count = ex.negative_count[idx]
        if len(ex.negative_games) == 0:
            return np.full(len(idx), NO_GAME, dtype=np.int64)
        pick = ex.negative_start[idx] + (self.rng.random(len(idx)) * count).astype(np.int64)
        games = ex.negative_games[np.clip(pick, 0, len(ex.negative_games) - 1)]
        return np.where(count > 0, games, NO_GAME)

    def _hard_negatives(self, games: np.ndarray, target: np.ndarray) -> HardNegatives:
        rows = self.catalog.rows(games)
        valid = (games != NO_GAME) & (rows >= 0) & (games != target)
        # invalid slots get catalog row 0 as a placeholder; they are masked in the loss
        items = self.catalog.items.take(np.where(valid, rows, 0))
        return HardNegatives(
            items=to_item_batch(items),
            game_idx=torch.from_numpy(np.where(valid, games, NO_GAME)),
            valid=torch.from_numpy(valid),
        )


class _Indices(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> int:
        return i


def make_loader(
    examples: TrainingExamples, collator: Collator, batch_size: int, generator: torch.Generator
) -> DataLoader:
    """Shuffling draws from `generator`: reseed it per epoch for resumable runs."""
    return DataLoader(
        _Indices(len(examples)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(examples) > batch_size,
        collate_fn=collator,
        generator=generator,
    )
