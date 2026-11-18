"""Stage 2, DataLoader / collate: training examples, hard negatives and batch assembly.

A training example is a kept positive interaction (`data.TrainRows`): the user's history as of
before the review is the query, the reviewed game is the target. Each example also carries:
  - an explicit hard negative: a game the same user reviewed negatively (training split);
  - a mined hard negative: refreshed at the start of each epoch (`evaluation.mine_hard_negatives`).
Other examples' targets in the batch are the in-batch negatives.

Target features: list features from the catalog (static per game), numerical features as of the
review. Collation is numpy; the batch becomes tensors at the end (`Batch.to(device)`).

A6 (item recency): the features say when they were measured (days since release + the game's
identity give the date away), so candidates of different dates in one softmax let the model
reject the "wrong date" instead of learning taste. With recency, every batch comes from one
time bucket (`time_buckets`: the evaluation's weeks, sparse old weeks merged), every candidate's
recency is as of the bucket's first day, and logQ is the target frequency within the bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from steam_training.contracts import OOV_ID, PADDING_ID
from steam_training.data import (
    NUM_RECENCY_FEATURES,
    Catalog,
    ItemFeatures,
    NegativeRows,
    Ragged,
    TrainRows,
    week_of_day,
    week_start_day,
)
from steam_training.model import Bag, ItemBatch, TensorTree

NO_GAME = -1
LANGUAGE_DROPOUT = 0.1


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
    # A6: day of each example's review, and (set by `train_model`) the day its batch's recency
    # is as of: the first day of its time bucket (without it: the review's day)
    day: np.ndarray | None = None
    as_of: np.ndarray | None = None
    # A6: log P(target | the example's time bucket), the logQ correction of bucketed batches
    log_q: np.ndarray | None = None
    # A7: language id of the user's previous review (OOV_ID unknown)
    language: np.ndarray | None = None

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
        day=train.day,
        language=train.language,
    )


def target_log_probs(target: np.ndarray, num_games: int) -> np.ndarray:
    """log P(game is a batch item) under uniform example sampling, for the logQ correction."""
    counts = np.bincount(target, minlength=num_games).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.log(counts / max(counts.sum(), 1.0)).astype(np.float32)


def time_buckets(day: np.ndarray, min_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """A6: ISO weeks (the weeks evaluation refreshes the catalog on, and the mart's); consecutive
    weeks are merged, oldest first, until a bucket has `min_rows` examples (the newest bucket
    keeps whatever is left). Returns each example's bucket and each bucket's first day (a
    Monday)."""
    week = week_of_day(day)
    weeks, inverse, counts = np.unique(week, return_inverse=True, return_counts=True)
    bucket_of_week = np.empty(len(weeks), np.int64)
    starts: list[int] = []
    size = min_rows
    for i, (w, n) in enumerate(zip(weeks.tolist(), counts.tolist(), strict=True)):
        if size >= min_rows:
            starts.append(int(week_start_day(w)))
            size = 0
        bucket_of_week[i] = len(starts) - 1
        size += n
    return bucket_of_week[inverse], np.asarray(starts, np.int64)


def bucket_log_probs(target: np.ndarray, bucket: np.ndarray) -> np.ndarray:
    """log P(game is a batch item | the example's bucket) under uniform sampling within it."""
    key = (bucket.astype(np.int64) << 32) | target.astype(np.int64)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    sizes = np.bincount(bucket)
    return np.log(counts[inverse] / sizes[bucket]).astype(np.float32)


def base_numeric(items: ItemFeatures) -> np.ndarray:
    return np.stack([items.is_free, items.is_free_known, items.reviews_ratio], axis=1)


def to_bag(ragged: Ragged) -> Bag:
    weights = torch.from_numpy(ragged.weights) if ragged.weights is not None else None
    return Bag(
        torch.from_numpy(ragged.values), torch.from_numpy(ragged.offsets[:-1].copy()), weights
    )


def to_item_batch(
    items: ItemFeatures, game_idx: np.ndarray | None = None, numeric: np.ndarray | None = None
) -> ItemBatch:
    if numeric is None:
        numeric = base_numeric(items)
        if items.recency is not None:  # A6: as of the catalog's day
            numeric = np.concatenate([numeric, items.recency], axis=1)
    return ItemBatch(
        game_idx=torch.from_numpy(items.game_idx if game_idx is None else game_idx),
        numeric=torch.from_numpy(numeric.astype(np.float32)),
        developers=to_bag(items.developers),
        publishers=to_bag(items.publishers),
        genres=to_bag(items.genres),
        categories=to_bag(items.categories),
        tags=to_bag(items.tags),
    )


@dataclass
class HardNegatives(TensorTree):
    items: ItemBatch  # one row per example (placeholder features where invalid)
    game_idx: torch.Tensor  # int64 [B]
    valid: torch.Tensor  # bool [B]


@dataclass
class Batch(TensorTree):
    history: torch.Tensor  # int64 [B, 5]
    items: ItemBatch  # targets
    target: torch.Tensor  # int64 [B] true game_idx (before id dropout)
    target_log_prob: torch.Tensor  # float32 [B]
    negatives: list[HardNegatives]

    # A7: language id of each row's user (OOV_ID unknown); None without USER_LANGUAGE
    language: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.target)


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
        hard_negative_id_dropout: float = 0.0,
        new_item_id_dropout: float | None = None,
        new_item_days: int = 90,
        recency_dropout: float = 0.0,
        explicit_negatives: bool = True,
        mined_negatives: bool = True,
        seed: int = 0,
    ) -> None:
        self.examples = examples
        self.catalog = catalog
        self.log_probs = log_probs
        self.history_dropout = history_dropout
        self.item_id_dropout = item_id_dropout
        self.hard_negative_id_dropout = hard_negative_id_dropout
        self.new_item_id_dropout = new_item_id_dropout
        self.new_item_days = new_item_days
        self.recency_dropout = recency_dropout
        self.explicit_negatives = explicit_negatives
        self.mined_negatives = mined_negatives
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # A6: recency as of each example's day (the target and its hard negatives)
        self.recency = catalog.recency if examples.day is not None else None

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
        day = None
        if self.recency is not None:
            as_of = ex.as_of if ex.as_of is not None else ex.day
            assert as_of is not None
            day = as_of[idx]
            numeric = np.concatenate([numeric, self.recency.features(target, day)], axis=1)
            numeric = self._drop_recency(numeric)
        if self.new_item_id_dropout is not None:  # new releases: the games unseen at serving
            young = self._young(target, day)
            dropped = np.where(young, self.rng.random(len(idx)) < self.new_item_id_dropout, dropped)
        items = to_item_batch(
            self.catalog.items.take(ex.catalog_row[idx]),
            game_idx=np.where(dropped, OOV_ID, target),
            numeric=numeric,
        )
        negatives = []
        if self.explicit_negatives:
            negatives.append(self._hard_negatives(self._sample_explicit(idx), target, day))
        if self.mined_negatives:
            negatives.append(self._hard_negatives(ex.mined[idx], target, day))
        return Batch(
            history=torch.from_numpy(history),
            items=items,
            target=torch.from_numpy(target),
            target_log_prob=torch.from_numpy(
                self.log_probs[target] if ex.log_q is None else ex.log_q[idx]
            ),
            language=self._language(idx),
            negatives=negatives,
        )

    def _drop_recency(self, numeric: np.ndarray) -> np.ndarray:
        """A6: zero an item's recency features w.p. `recency_dropout` (all four, so the
        release reads as unknown), so taste has to carry the score and "trending this week"
        can't win alone. Serving always has them."""
        if not self.recency_dropout:
            return numeric
        dropped = self.rng.random(len(numeric)) < self.recency_dropout
        numeric = numeric.copy()
        numeric[dropped, -NUM_RECENCY_FEATURES:] = 0.0
        return numeric

    def _young(self, games: np.ndarray, day: np.ndarray | None) -> np.ndarray:
        """Bool per game: released at most `new_item_days` before `day` (known release only)."""
        assert self.recency is not None and day is not None, "NEW_ITEM_ID_DROPOUT needs A6"
        release_day = self.recency.release_day
        inside = (games >= 0) & (games < len(release_day))
        release = np.where(inside, release_day[np.clip(games, 0, len(release_day) - 1)], -1)
        age = np.asarray(day, np.int64) - release
        return (release >= 0) & (age >= 0) & (age <= self.new_item_days)

    def _negative_id_dropout(self, games: np.ndarray, day: np.ndarray | None) -> np.ndarray:
        """Bool per hard negative: its id is replaced by OOV (same rates as the targets)."""
        if not self.hard_negative_id_dropout and self.new_item_id_dropout is None:
            return np.zeros(len(games), bool)
        dropped = self.rng.random(len(games)) < self.hard_negative_id_dropout
        if self.new_item_id_dropout is not None:
            young = self._young(games, day)
            dropped = np.where(
                young, self.rng.random(len(games)) < self.new_item_id_dropout, dropped
            )
        return dropped

    def _language(self, idx: np.ndarray) -> torch.Tensor | None:
        """A7: the user's language, dropped to unknown w.p. LANGUAGE_DROPOUT (the online path
        and new users don't know it)."""
        if self.examples.language is None:
            return None
        language = self.examples.language[idx].astype(np.int64)
        dropped = self.rng.random(len(idx)) < LANGUAGE_DROPOUT
        return torch.from_numpy(np.where(dropped, OOV_ID, language))

    def _sample_explicit(self, idx: np.ndarray) -> np.ndarray:
        ex = self.examples
        count = ex.negative_count[idx]
        if len(ex.negative_games) == 0:
            return np.full(len(idx), NO_GAME, dtype=np.int64)
        pick = ex.negative_start[idx] + (self.rng.random(len(idx)) * count).astype(np.int64)
        games = ex.negative_games[np.clip(pick, 0, len(ex.negative_games) - 1)]
        return np.where(count > 0, games, NO_GAME)

    def _hard_negatives(
        self, games: np.ndarray, target: np.ndarray, day: np.ndarray | None = None
    ) -> HardNegatives:
        rows = self.catalog.rows(games)
        valid = (games != NO_GAME) & (rows >= 0) & (games != target)
        # invalid slots get catalog row 0 as a placeholder; they are masked in the loss
        items = self.catalog.items.take(np.where(valid, rows, 0))
        numeric = None
        if self.recency is not None:  # A6: as of the example's day, like its target
            numeric = np.concatenate(
                [base_numeric(items), self.recency.features(items.game_idx, day)], axis=1
            )
            numeric = self._drop_recency(numeric)
        dropped = self._negative_id_dropout(np.where(valid, games, NO_GAME), day)
        return HardNegatives(
            items=to_item_batch(
                items, game_idx=np.where(dropped, OOV_ID, items.game_idx), numeric=numeric
            ),
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


class BucketBatchSampler(Sampler[list[int]]):
    """A6: batches of one time bucket each. Every epoch, each bucket's examples are shuffled and
    split into round(n / batch_size) near-equal batches (no tiny batch is a full step), and the
    batches of every bucket are shuffled together. Draws from `generator` (reseeded per epoch)."""

    def __init__(self, bucket: np.ndarray, batch_size: int, generator: torch.Generator) -> None:
        order = np.argsort(bucket, kind="stable").astype(np.int32)
        bounds = np.searchsorted(bucket[order], np.arange(int(bucket.max()) + 2))
        self.groups = [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]
        self.groups = [g for g in self.groups if len(g)]
        self.splits = [max(1, round(len(g) / batch_size)) for g in self.groups]
        self.generator = generator

    def __len__(self) -> int:
        return sum(self.splits)

    def __iter__(self):
        batches: list[np.ndarray] = []
        for group, splits in zip(self.groups, self.splits, strict=True):
            shuffled = group[torch.randperm(len(group), generator=self.generator).numpy()]
            batches.extend(np.array_split(shuffled, splits))
        for i in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[i].tolist()


def make_loader(
    examples: TrainingExamples,
    collator: Collator,
    batch_size: int,
    generator: torch.Generator,
    pin_memory: bool = False,
    bucket: np.ndarray | None = None,
) -> DataLoader:
    """Shuffling draws from `generator`: reseed it per epoch for resumable runs. `pin_memory`
    (CUDA): batches come pinned, so `Batch.to(device, non_blocking=True)` copies them
    asynchronously instead of one blocking copy per tensor (~60 per batch). `bucket` (A6): each
    batch from one time bucket (`BucketBatchSampler`)."""
    if bucket is not None:
        return DataLoader(
            _Indices(len(examples)),
            batch_sampler=BucketBatchSampler(bucket, batch_size, generator),
            collate_fn=collator,
            pin_memory=pin_memory,
        )
    return DataLoader(
        _Indices(len(examples)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(examples) > batch_size,
        collate_fn=collator,
        generator=generator,
        pin_memory=pin_memory,
    )
