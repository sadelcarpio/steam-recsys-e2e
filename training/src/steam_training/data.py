"""Stage 1, dataset + split: stream the Iceberg marts and keep only what training needs.

Memory scales with the rows *kept*, not with the table: `interactions` is read in Arrow batches
(pinned to one snapshot) in three passes:
  1. `timestamp`, `is_positive` only -> temporal cutoff (last ~10% of rows = validation) and the
     sampling rate that caps the validation rows at `eval_max_rows`;
  2. every column, filtered batch by batch:
       - training positives: all warm (non-empty history) rows + `cold_row_fraction` of the cold
         ones -> history, target game, as-of numerical features, user;
       - training negatives (`is_positive = false`): user + game only (explicit hard negatives);
       - validation positives, uniformly sampled -> history, target, warm flag;
       - positive counts per game in the training split (popularity baseline + logQ);
  3. `game_features` rows before the cutoff, reduced to the latest row per game (the catalog),
     joined with each game's current tags (`game_tags`, `with_tags`).
Game list features (developers, publishers, genres, categories, tags) come from the catalog: they
are static per game, only `game_reviews_ratio` / `game_is_free` are kept per row (as of the review).

Variable-length id lists are CSR arrays (`Ragged`): batches are gathered with vectorised numpy
and fed straight into `nn.EmbeddingBag` (values + offsets).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from steam_training.contracts import PADDING_ID, USER_HISTORY_LENGTH, SplitInfo, VocabSizes

log = logging.getLogger(__name__)

ITEM_COLUMNS = [
    "game_idx",
    "game_is_free",
    "game_developers",
    "game_publishers",
    "game_genres",
    "game_categories",
    "game_reviews_ratio",
]
INTERACTION_COLUMNS = [
    "timestamp",
    "user_id",
    "is_positive",
    "games_reviewed_positive",
    "game_idx",
    "game_is_free",
    "game_reviews_ratio",
]
GAME_FEATURE_COLUMNS = ["timestamp", *ITEM_COLUMNS]
GAME_TAG_COLUMNS = ["game_idx", "game_tags", "game_tag_weights"]
# lookup table -> (id column, VocabSizes field)
LOOKUPS = {
    "lkp_games": ("game_idx", "games"),
    "lkp_developers": ("id", "developers"),
    "lkp_publishers": ("id", "publishers"),
    "lkp_genres": ("id", "genres"),
    "lkp_categories": ("id", "categories"),
}
TAG_LOOKUPS = {"lkp_tags": ("id", "tags")}


# ---- sources -------------------------------------------------------------------------------


class TableSource(Protocol):
    def snapshot_id(self, table: str) -> int | None:
        """Current snapshot of a mart; every read of one load is pinned to it."""
        ...

    def batches(
        self, table: str, columns: list[str], snapshot_id: int | None
    ) -> Iterator[pa.RecordBatch]:
        """Stream the selected columns of a mart at `snapshot_id`."""
        ...


class IcebergSource:
    """Reads the marts through the Glue catalog, straight from S3 (no Athena, no scan cost)."""

    def __init__(self, database: str, region: str) -> None:
        from pyiceberg.catalog import load_catalog

        self._database = database
        self._catalog = load_catalog("glue", type="glue", **{"glue.region": region})
        self._tables: dict = {}

    def _table(self, name: str):
        if name not in self._tables:
            self._tables[name] = self._catalog.load_table((self._database, name))
        return self._tables[name]

    def snapshot_id(self, table: str) -> int | None:
        snapshot = self._table(table).current_snapshot()
        return snapshot.snapshot_id if snapshot else None

    def batches(
        self, table: str, columns: list[str], snapshot_id: int | None
    ) -> Iterator[pa.RecordBatch]:
        scan = self._table(table).scan(selected_fields=tuple(columns), snapshot_id=snapshot_id)
        yield from scan.to_arrow_batch_reader()


# ---- in-memory layout ----------------------------------------------------------------------


@dataclass(frozen=True)
class Ragged:
    """CSR list-of-ints: row i is values[offsets[i]:offsets[i + 1]], with optional per-value
    weights (aligned with `values`)."""

    values: np.ndarray  # int64
    offsets: np.ndarray  # int64, len = rows + 1
    weights: np.ndarray | None = None  # float32, len = len(values)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def from_arrow(cls, column: pa.ChunkedArray | pa.Array) -> Ragged:
        """Null lists become empty lists."""
        lengths = pc.fill_null(pc.list_value_length(column), 0).to_numpy(zero_copy_only=False)
        values = pc.list_flatten(column)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return cls(np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64), offsets)

    @classmethod
    def empty(cls, rows: int) -> Ragged:
        return cls(np.zeros(0, dtype=np.int64), np.zeros(rows + 1, dtype=np.int64))

    @classmethod
    def from_lists(cls, rows: list[list[int]]) -> Ragged:
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum([len(r) for r in rows], out=offsets[1:])
        values = np.fromiter((v for r in rows for v in r), dtype=np.int64, count=int(offsets[-1]))
        return cls(values, offsets)

    def take(self, index: np.ndarray) -> Ragged:
        starts = self.offsets[index]
        lengths = self.offsets[index + 1] - starts
        offsets = np.zeros(len(index) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        # position of each output value in `values`: its row start + rank inside the row
        within = np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], lengths)
        source = np.repeat(starts, lengths) + within
        weights = self.weights[source] if self.weights is not None else None
        return Ragged(self.values[source], offsets, weights)

    def row(self, i: int) -> list[int]:
        return self.values[self.offsets[i] : self.offsets[i + 1]].tolist()


@dataclass(frozen=True)
class ItemFeatures:
    game_idx: np.ndarray  # int64
    is_free: np.ndarray  # float32, 0 when unknown
    is_free_known: np.ndarray  # float32
    reviews_ratio: np.ndarray  # float32
    developers: Ragged
    publishers: Ragged
    genres: Ragged
    categories: Ragged
    # weighted: each row's weights sum to 1 (empty row: no tags); None -> no tags at all
    tags: Ragged = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tags is None:
            object.__setattr__(self, "tags", Ragged.empty(len(self.game_idx)))

    def __len__(self) -> int:
        return len(self.game_idx)

    @classmethod
    def from_arrow(cls, table: pa.Table | pa.RecordBatch) -> ItemFeatures:
        is_free, is_free_known, reviews_ratio = _numeric_features(table)
        return cls(
            game_idx=_int64(table["game_idx"]),
            is_free=is_free,
            is_free_known=is_free_known,
            reviews_ratio=reviews_ratio,
            developers=Ragged.from_arrow(table["game_developers"]),
            publishers=Ragged.from_arrow(table["game_publishers"]),
            genres=Ragged.from_arrow(table["game_genres"]),
            categories=Ragged.from_arrow(table["game_categories"]),
            tags=_weighted_tags(table),
        )

    def take(self, index: np.ndarray) -> ItemFeatures:
        return ItemFeatures(
            game_idx=self.game_idx[index],
            is_free=self.is_free[index],
            is_free_known=self.is_free_known[index],
            reviews_ratio=self.reviews_ratio[index],
            developers=self.developers.take(index),
            publishers=self.publishers.take(index),
            genres=self.genres.take(index),
            categories=self.categories.take(index),
            tags=self.tags.take(index),
        )


@dataclass(frozen=True)
class Catalog:
    """One row per game: its features as of a point in time (the ranking corpus)."""

    items: ItemFeatures
    row_of: np.ndarray  # int64 [vocab.games]: catalog row of each game_idx, -1 when absent

    def __len__(self) -> int:
        return len(self.items)

    def rows(self, game_idx: np.ndarray) -> np.ndarray:
        """Catalog rows of `game_idx` (-1 for games outside the catalog / vocabulary)."""
        inside = (game_idx >= 0) & (game_idx < len(self.row_of))
        return np.where(inside, self.row_of[np.clip(game_idx, 0, len(self.row_of) - 1)], -1)


@dataclass(frozen=True)
class TrainRows:
    """Kept positive training rows (all warm + a sample of cold ones)."""

    user_id: np.ndarray  # int64
    history: np.ndarray  # int64 [n, 5], most recent first, 0 = padding
    target: np.ndarray  # int64 game_idx
    is_free: np.ndarray  # float32 (as of the review)
    is_free_known: np.ndarray  # float32
    reviews_ratio: np.ndarray  # float32 (as of the review)

    def __len__(self) -> int:
        return len(self.target)

    def take(self, index: np.ndarray) -> TrainRows:
        return TrainRows(**{name: getattr(self, name)[index] for name in _fields(self)})


@dataclass(frozen=True)
class NegativeRows:
    """Negative reviews of the training split (explicit hard negatives)."""

    user_id: np.ndarray  # int64
    game_idx: np.ndarray  # int64

    def __len__(self) -> int:
        return len(self.game_idx)


@dataclass(frozen=True)
class EvalRows:
    """Sampled positive validation rows."""

    history: np.ndarray  # int64 [n, 5]
    target: np.ndarray  # int64 [n]
    is_warm: np.ndarray  # bool [n]

    def __len__(self) -> int:
        return len(self.target)

    def sample(self, max_rows: int, seed: int = 0) -> EvalRows:
        if len(self) <= max_rows:
            return self
        index = np.sort(np.random.default_rng(seed).choice(len(self), max_rows, replace=False))
        return EvalRows(self.history[index], self.target[index], self.is_warm[index])


@dataclass(frozen=True)
class TrainingData:
    split: SplitInfo
    vocab: VocabSizes
    snapshots: dict[str, int | None]
    catalog: Catalog
    validation: EvalRows
    # positive reviews per game_idx in the training split (all rows, before cold sampling)
    train_positive_counts: np.ndarray
    # None when loaded for evaluation only
    train: TrainRows | None
    negatives: NegativeRows | None


def load_training_data(
    source: TableSource,
    *,
    validation_fraction: float,
    cold_row_fraction: float,
    eval_max_rows: int,
    seed: int,
    cutoff: datetime | None = None,
    with_train_rows: bool = True,
    with_tags: bool = False,
) -> TrainingData:
    """Stream the marts (see module docstring). `cutoff` given: split there (promotion) instead
    of at the `validation_fraction` quantile. `with_tags`: read `lkp_tags` / `game_tags` (the
    vocabulary gets `tags`, the catalog each game's tags)."""
    lookups = {**LOOKUPS, **(TAG_LOOKUPS if with_tags else {})}
    tables = [*lookups, "interactions", "game_features", *(["game_tags"] if with_tags else [])]
    snapshots = {table: source.snapshot_id(table) for table in tables}
    vocab = _vocab_sizes(source, snapshots, lookups)

    # pass 1: cutoff + validation sampling rate
    timestamps, positive = [], []
    for batch in source.batches(
        "interactions", ["timestamp", "is_positive"], snapshots["interactions"]
    ):
        timestamps.append(_timestamps(batch["timestamp"]))
        positive.append(_bool(batch["is_positive"]))
    all_ts = np.concatenate(timestamps) if timestamps else np.array([], dtype="datetime64[us]")
    all_positive = np.concatenate(positive) if positive else np.array([], dtype=bool)
    if cutoff is None:
        cutoff = temporal_cutoff(all_ts, validation_fraction)
    is_validation = all_ts >= np.datetime64(cutoff, "us")
    validation_positives = int((is_validation & all_positive).sum())
    eval_rate = min(1.0, eval_max_rows / max(validation_positives, 1))
    split = SplitInfo(
        cutoff=cutoff,
        train_rows=int((~is_validation).sum()),
        validation_rows=int(is_validation.sum()),
    )
    del timestamps, positive, all_ts, all_positive, is_validation
    log.info(
        "split at %s: %d train / %d validation rows (%d validation positives, eval rate %.3f)",
        cutoff,
        split.train_rows,
        split.validation_rows,
        validation_positives,
        eval_rate,
    )

    # pass 2: filter + sample batch by batch
    rng = np.random.default_rng(seed)
    cutoff_us = np.datetime64(cutoff, "us")
    counts = np.zeros(vocab.games, dtype=np.int64)
    train_parts: list[TrainRows] = []
    negative_parts: list[NegativeRows] = []
    eval_parts: list[EvalRows] = []
    for batch in source.batches("interactions", INTERACTION_COLUMNS, snapshots["interactions"]):
        game = _int64(pc.fill_null(batch["game_idx"], -1))
        known = (game >= 0) & (game < vocab.games)
        is_positive = _bool(batch["is_positive"]) & known
        is_val = _timestamps(batch["timestamp"]) >= cutoff_us
        history = pad_history(Ragged.from_arrow(batch["games_reviewed_positive"]))
        warm = history[:, 0] != PADDING_ID
        counts += np.bincount(game[is_positive & ~is_val], minlength=vocab.games)

        keep_eval = np.flatnonzero(is_positive & is_val & (rng.random(len(game)) < eval_rate))
        eval_parts.append(EvalRows(history[keep_eval], game[keep_eval], warm[keep_eval]))
        if not with_train_rows:
            continue
        sampled = warm | (rng.random(len(game)) < cold_row_fraction)
        keep_train = np.flatnonzero(is_positive & ~is_val & sampled)
        is_free, is_free_known, reviews_ratio = _numeric_features(batch)
        user_id = _int64(batch["user_id"])
        train_parts.append(
            TrainRows(
                user_id=user_id[keep_train],
                history=history[keep_train],
                target=game[keep_train],
                is_free=is_free[keep_train],
                is_free_known=is_free_known[keep_train],
                reviews_ratio=reviews_ratio[keep_train],
            )
        )
        keep_negative = np.flatnonzero(~_bool(batch["is_positive"]) & known & ~is_val)
        negative_parts.append(NegativeRows(user_id[keep_negative], game[keep_negative]))

    # pass 3: catalog as of the cutoff (+ the games' current tags)
    tags = (
        source.batches("game_tags", GAME_TAG_COLUMNS, snapshots["game_tags"]) if with_tags else None
    )
    catalog = catalog_at(
        source.batches("game_features", GAME_FEATURE_COLUMNS, snapshots["game_features"]),
        cutoff,
        vocab.games,
        tags,
    )
    train = _concat(TrainRows, train_parts) if with_train_rows else None
    if train is not None:
        # targets always have a catalog row (1970-01-01 game_features row); guard anyway
        train = train.take(np.flatnonzero(catalog.rows(train.target) >= 0))
    data = TrainingData(
        split=split,
        vocab=vocab,
        snapshots=snapshots,
        catalog=catalog,
        validation=_concat(EvalRows, eval_parts),
        train_positive_counts=counts,
        train=train,
        negatives=_concat(NegativeRows, negative_parts) if with_train_rows else None,
    )
    log.info(
        "kept %s training positives, %s negatives, %d validation rows, %d catalog games",
        len(train) if train is not None else "no",
        len(data.negatives) if data.negatives is not None else "no",
        len(data.validation),
        len(catalog),
    )
    return data


# ---- split + catalog -----------------------------------------------------------------------


def temporal_cutoff(timestamps: np.ndarray, validation_fraction: float) -> datetime:
    """Timestamp such that about `validation_fraction` of the rows are at or after it."""
    if len(timestamps) == 0:
        raise ValueError("no interactions to split")
    position = min(int(len(timestamps) * (1 - validation_fraction)), len(timestamps) - 1)
    value = np.partition(timestamps, position)[position]
    return value.astype("datetime64[us]").item()


def catalog_at(
    batches: Iterable[pa.RecordBatch],
    cutoff: datetime,
    num_games: int,
    tag_batches: Iterable[pa.RecordBatch] | None = None,
) -> Catalog:
    """Latest game_features row per game strictly before `cutoff` (what batch inference at
    `cutoff` would see), reduced while streaming. Every game has a 1970-01-01 row, so every known
    game is included. `tag_batches` (the `game_tags` mart) adds each game's current tags."""
    latest = latest_game_rows(batches, num_games, cutoff)
    if tag_batches is not None:
        latest = attach_game_tags(latest, tag_batches, num_games)
    return catalog_from_table(latest, num_games)


def attach_game_tags(
    latest: pa.Table, tag_batches: Iterable[pa.RecordBatch], num_games: int
) -> pa.Table:
    """`latest` (one row per game) plus the `game_tags` / `game_tag_weights` columns of the
    `game_tags` mart, joined by `game_idx` (null for games without tags: an empty bag). Tags are
    static per game, like genres: the current tags, whatever the catalog's cutoff."""
    batches = list(tag_batches)
    schema = pa.schema(
        [
            ("game_idx", pa.int64()),
            ("game_tags", pa.list_(pa.int64())),
            ("game_tag_weights", pa.list_(pa.float64())),
        ]
    )
    tags = (
        pa.Table.from_batches(batches).select(GAME_TAG_COLUMNS).cast(schema)
        if batches
        else schema.empty_table()
    )
    row_of = np.full(num_games, -1, dtype=np.int64)
    tag_game = _int64(pc.fill_null(tags["game_idx"], num_games))
    inside = (tag_game >= 0) & (tag_game < num_games)
    row_of[tag_game[inside]] = np.flatnonzero(inside)
    game = _int64(latest["game_idx"])
    rows = np.where((game >= 0) & (game < num_games), row_of[np.clip(game, 0, num_games - 1)], -1)
    index = pa.array(rows, mask=rows < 0)
    for name in ("game_tags", "game_tag_weights"):
        if name in latest.column_names:
            latest = latest.drop_columns(name)
        latest = latest.append_column(name, tags[name].take(index))
    return latest


def latest_game_rows(
    batches: Iterable[pa.RecordBatch], num_games: int, cutoff: datetime | None = None
) -> pa.Table:
    """Latest row per `game_idx` < `num_games` (strictly before `cutoff` when given), reduced
    batch by batch. Keeps every selected column (inference also reads names / game ids)."""
    cutoff_us = np.datetime64(cutoff, "us") if cutoff is not None else None
    latest: pa.Table | None = None
    for batch in batches:
        keep = _int64(pc.fill_null(batch["game_idx"], num_games)) < num_games
        if cutoff_us is not None:
            keep &= _timestamps(batch["timestamp"]) < cutoff_us
        rows = pa.Table.from_batches([batch]).filter(pa.array(keep))
        latest = _latest_per_game(rows if latest is None else pa.concat_tables([latest, rows]))
    if latest is None or latest.num_rows == 0:
        raise ValueError(f"no game_features rows before {cutoff}")
    return latest


def catalog_from_table(latest: pa.Table, num_games: int) -> Catalog:
    """Catalog over one row per game (`latest_game_rows`), in the table's row order."""
    items = ItemFeatures.from_arrow(latest)
    row_of = np.full(num_games, -1, dtype=np.int64)
    row_of[items.game_idx] = np.arange(len(items))
    return Catalog(items=items, row_of=row_of)


def _latest_per_game(table: pa.Table) -> pa.Table:
    game_idx = _int64(table["game_idx"])
    order = np.lexsort((_timestamps(table["timestamp"]), game_idx))
    ordered = game_idx[order]
    is_last = np.ones(len(order), dtype=bool)
    is_last[:-1] = ordered[:-1] != ordered[1:]
    return table.take(pa.array(order[is_last]))


# ---- helpers -------------------------------------------------------------------------------


def _vocab_sizes(
    source: TableSource,
    snapshots: dict[str, int | None],
    lookups: dict[str, tuple[str, str]] = LOOKUPS,
) -> VocabSizes:
    sizes: dict[str, int] = {}
    for table, (column, field) in lookups.items():
        max_id = 0
        for batch in source.batches(table, [column], snapshots[table]):
            if batch.num_rows:
                max_id = max(max_id, int(pc.max(batch[column]).as_py() or 0))
        # at least the reserved ids + 1 so every embedding table has a real row
        sizes[field] = max(max_id + 1, 3)
    return VocabSizes(**sizes)


def _numeric_features(
    table: pa.Table | pa.RecordBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_free = table["game_is_free"]
    return (
        pc.fill_null(is_free, False).to_numpy(zero_copy_only=False).astype(np.float32),
        pc.is_valid(is_free).to_numpy(zero_copy_only=False).astype(np.float32),
        pc.fill_null(table["game_reviews_ratio"], 0.5)
        .to_numpy(zero_copy_only=False)
        .astype(np.float32),
    )


def _weighted_tags(table: pa.Table | pa.RecordBatch) -> Ragged:
    """Tags of each row with their weights normalized to sum 1 (uniform when they sum to 0);
    empty rows when the table has no tag columns."""
    if "game_tags" not in table.column_names:
        return Ragged.empty(table.num_rows)
    tags = Ragged.from_arrow(table["game_tags"])
    raw = pc.list_flatten(table["game_tag_weights"]).to_numpy(zero_copy_only=False)
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0).clip(min=0)
    if len(raw) != len(tags.values):
        raise ValueError("game_tags and game_tag_weights are not aligned")
    lengths = np.diff(tags.offsets)
    owner = np.repeat(np.arange(len(lengths)), lengths)
    totals = np.bincount(owner, weights=raw, minlength=len(lengths))
    uniform = 1.0 / np.maximum(lengths, 1)
    weights = np.where(totals[owner] > 0, raw / np.maximum(totals[owner], 1e-12), uniform[owner])
    return Ragged(tags.values, tags.offsets, weights.astype(np.float32))


def _fields(obj) -> list[str]:
    return list(obj.__dataclass_fields__)


def _concat[T](cls: type[T], parts: list[T]) -> T:
    names = list(cls.__dataclass_fields__)
    if not parts:
        raise ValueError(f"no {cls.__name__} rows")
    return cls(**{name: np.concatenate([getattr(p, name) for p in parts]) for name in names})


def _int64(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _bool(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return np.asarray(pc.fill_null(column, False).to_numpy(zero_copy_only=False), dtype=bool)


def _timestamps(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    column = pc.cast(column, pa.timestamp("us"))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype="datetime64[us]")


def pad_history(history: Ragged) -> np.ndarray:
    """Fixed [rows, USER_HISTORY_LENGTH] matrix (the mart already pads, this guards nulls)."""
    rows = len(history)
    out = np.full((rows, USER_HISTORY_LENGTH), PADDING_ID, dtype=np.int64)
    lengths = np.minimum(np.diff(history.offsets), USER_HISTORY_LENGTH)
    column = np.arange(USER_HISTORY_LENGTH)
    mask = column[None, :] < lengths[:, None]
    source = history.offsets[:-1, None] + column[None, :]
    out[mask] = history.values[source[mask]]
    return out
