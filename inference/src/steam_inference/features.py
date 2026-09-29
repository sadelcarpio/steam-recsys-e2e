"""Stage 1: the current state of every user and game, read from the Iceberg marts.

`game_features` and `user_features` are the source of truth: the latest row per game / user is
its current state. `interactions` only adds each user's reviewed games (excluded from the
recommendations), review counts (who gets reranked) and the recent positive reviews per game
(the popularity fallback served to users without recommendations). Every read of one run is
pinned to the snapshot current at its start, and marts are streamed in Arrow batches, reduced
as they arrive.

The catalog covers every current game, including games newer than the model: ids beyond the
model's vocabularies map to OOV inside the model, so a new game is scored from its content
features. Each game's Steam user tags (`game_tags`, spec 8) are joined when the mart exists: a
model trained with tags needs them, and the prompt / adult filter use their names. Each game's
release status comes from `game_details` (spec 10: unreleased games are never published).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.exceptions import NoSuchTableError
from steam_training.contracts import USER_HISTORY_LENGTH
from steam_training.data import (
    GAME_FEATURE_COLUMNS,
    GAME_TAG_COLUMNS,
    Catalog,
    Ragged,
    TableSource,
    attach_game_tags,
    catalog_from_table,
    latest_game_rows,
    pad_history,
)

from steam_inference.contracts import clean_text

log = logging.getLogger(__name__)

TABLES = [
    "lkp_games",
    "lkp_genres",
    "lkp_developers",
    "game_features",
    "user_features",
    "interactions",
]
# Steam user tags (spec 8): used when both exist.
TAG_TABLES = ["lkp_tags", "game_tags"]
# Tags listed per game in the rerank prompt (the heaviest ones).
PROMPT_TAGS = 5
# Rows buffered before the latest-row-per-user reduction runs again.
REDUCE_ROWS = 1_000_000


@dataclass(frozen=True)
class Games:
    catalog: Catalog  # current features of every game (the ranking corpus)
    game_id: np.ndarray  # int64 per catalog row: Steam appid
    name: np.ndarray  # object (str) per catalog row
    genre_names: np.ndarray  # object (str) indexed by genre id
    developer_names: np.ndarray  # object (str) indexed by developer id
    # object (str) per catalog row: Steam short description, truncated ("" = none / not loaded)
    description: np.ndarray
    # object (str) indexed by tag id; empty when the tags were not loaded
    tag_names: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    # bool per catalog row: unreleased at its latest scrape (spec 10); None = all released
    coming_soon: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.catalog)

    def describe(self, row: int) -> str:
        """One-line description of a catalog row for the LLM prompt."""
        items = self.catalog.items
        genres = _names(self.genre_names, items.genres.row(row))
        developers = _names(self.developer_names, items.developers.row(row))[:2]
        parts = [self.name[row]]
        if genres:
            parts.append(", ".join(genres))
        tags = _names(self.tag_names, items.tags.row(row))[:PROMPT_TAGS]
        if tags:
            parts.append("tags: " + ", ".join(tags))
        if developers:
            parts.append("by " + ", ".join(developers))
        if items.is_free_known[row]:
            parts.append("free to play" if items.is_free[row] else "paid")
        parts.append(f"{items.reviews_ratio[row]:.0%} positive reviews")
        if self.description[row]:
            parts.append(self.description[row])
        return " | ".join(parts)


@dataclass(frozen=True)
class Users:
    """Latest user_features row per user, sorted by user_id, with their reviews
    (`review_start` / `review_count` into `Reviews`)."""

    user_id: np.ndarray  # int64 [n]
    history: np.ndarray  # int64 [n, 5] last positively reviewed games, most recent first
    review_start: np.ndarray  # int64 [n]
    review_count: np.ndarray  # int64 [n]

    def __len__(self) -> int:
        return len(self.user_id)

    def take(self, index: np.ndarray) -> Users:
        return Users(
            self.user_id[index],
            self.history[index],
            self.review_start[index],
            self.review_count[index],
        )


@dataclass(frozen=True)
class Reviews:
    """Reviewed game of every review, grouped by user (ascending), newest first within a user."""

    game_idx: np.ndarray  # int64

    def of(self, users: Users) -> Ragged:
        """Reviewed games of `users` as CSR rows."""
        offsets = np.zeros(len(users) + 1, dtype=np.int64)
        np.cumsum(users.review_count, out=offsets[1:])
        within = np.arange(offsets[-1], dtype=np.int64) - np.repeat(
            offsets[:-1], users.review_count
        )
        source = np.repeat(users.review_start, users.review_count) + within
        return Ragged(self.game_idx[source], offsets)


@dataclass(frozen=True)
class InferenceData:
    games: Games
    users: Users
    reviews: Reviews
    # int64 per game_idx: positive reviews in the last `popular_window_days` of reviews
    popular_counts: np.ndarray
    snapshots: dict[str, int | None]


def load_inference_data(
    source: TableSource,
    *,
    max_users: int = 0,
    popular_window_days: int = 90,
    description_chars: int = 0,
    require_tags: bool = False,
) -> InferenceData:
    """`description_chars` > 0 also loads the games' short descriptions (for the rerank
    prompt), truncated to that many characters. `require_tags`: fail when the tag marts are
    missing (the model was trained with tags)."""
    snapshots = {table: source.snapshot_id(table) for table in TABLES}
    snapshots |= _optional_snapshots(source, TAG_TABLES)
    if require_tags and not set(TAG_TABLES) <= set(snapshots):
        raise RuntimeError(f"the model uses Steam tags but {TAG_TABLES} are missing: run the ETL")
    games = load_games(source, snapshots, description_chars)
    review_users, reviews, popular_counts = _load_reviews(
        source, snapshots["interactions"], popular_window_days
    )
    users = _latest_users(
        source.batches(
            "user_features",
            ["user_id", "timestamp", "games_reviewed_positive"],
            snapshots["user_features"],
        )
    )
    start = np.searchsorted(review_users, users.user_id, side="left")
    count = np.searchsorted(review_users, users.user_id, side="right") - start
    users = Users(users.user_id, users.history, start, count)
    if max_users and len(users) > max_users:
        # the most active users (ties: lowest user id), kept in user id order
        keep = np.lexsort((users.user_id, -users.review_count))[:max_users]
        users = users.take(np.sort(keep))
    log.info(
        "loaded %d users, %d reviews, %d catalog games (snapshots %s)",
        len(users),
        len(reviews.game_idx),
        len(games),
        snapshots,
    )
    return InferenceData(
        games=games,
        users=users,
        reviews=reviews,
        popular_counts=popular_counts,
        snapshots=snapshots,
    )


def load_games(
    source: TableSource, snapshots: dict[str, int | None], description_chars: int = 0
) -> Games:
    num_games = _max_id(source, "lkp_games", "game_idx", snapshots["lkp_games"]) + 1
    latest = latest_game_rows(
        source.batches(
            "game_features",
            [*GAME_FEATURE_COLUMNS, "game_id", "game_name"],
            snapshots["game_features"],
        ),
        num_games,
    )
    tag_names = np.zeros(0, dtype=object)
    if set(TAG_TABLES) <= set(snapshots):
        tags = source.batches("game_tags", GAME_TAG_COLUMNS, snapshots["game_tags"])
        latest = attach_game_tags(latest, tags, num_games)
        tag_names = _lookup_names(source, "lkp_tags", snapshots["lkp_tags"])
    else:
        log.warning("marts %s not found: games without Steam tags", TAG_TABLES)
    game_id = _int64(latest["game_id"])
    details = _game_details(source, game_id, description_chars)
    return Games(
        catalog=catalog_from_table(latest, num_games),
        game_id=game_id,
        name=np.asarray(pc.fill_null(latest["game_name"], "").to_pylist(), dtype=object),
        genre_names=_lookup_names(source, "lkp_genres", snapshots["lkp_genres"]),
        developer_names=_lookup_names(source, "lkp_developers", snapshots["lkp_developers"]),
        description=details.description,
        tag_names=tag_names,
        coming_soon=details.coming_soon,
    )


def _optional_snapshots(source: TableSource, tables: list[str]) -> dict[str, int | None]:
    snapshots = {}
    for table in tables:
        try:
            snapshots[table] = source.snapshot_id(table)
        except NoSuchTableError:
            continue
    return snapshots


@dataclass(frozen=True)
class _Details:
    description: np.ndarray  # object (str) per catalog row
    coming_soon: np.ndarray  # bool per catalog row


def _game_details(source: TableSource, game_id: np.ndarray, max_chars: int) -> _Details:
    """Per catalog row, from the mart `game_details`: the short description (cleaned,
    truncated; "" when `max_chars` is 0, the game has none, or the mart is missing) and whether
    the game is unreleased (false when the mart is missing)."""
    description = np.full(len(game_id), "", dtype=object)
    coming_soon = np.zeros(len(game_id), dtype=bool)
    out = _Details(description, coming_soon)
    try:
        snapshot_id = source.snapshot_id("game_details")
    except NoSuchTableError:
        log.warning("mart game_details not found: no descriptions, every game counted released")
        return out
    columns = ["game_id", "game_coming_soon"]
    if max_chars:
        columns.append("game_short_description")
    row_of = {int(g): i for i, g in enumerate(game_id)}
    found = 0
    for batch in source.batches("game_details", columns, snapshot_id):
        texts = batch["game_short_description"].to_pylist() if max_chars else None
        unreleased = batch["game_coming_soon"].to_pylist()
        for i, gid in enumerate(batch["game_id"].to_pylist()):
            row = row_of.get(gid)
            if row is None:
                continue
            coming_soon[row] = bool(unreleased[i])
            if texts is not None and (text := clean_text(texts[i])):
                description[row] = truncate(text, max_chars)
                found += 1
    if max_chars:
        log.info("short descriptions for %d of %d catalog games", found, len(game_id))
    log.info("unreleased games in the catalog: %d", int(coming_soon.sum()))
    return out


def truncate(text: str, max_chars: int) -> str:
    """At most `max_chars` characters, cut at a word boundary with an ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "…"


def _load_reviews(
    source: TableSource, snapshot_id: int | None, popular_window_days: int
) -> tuple[np.ndarray, Reviews, np.ndarray]:
    """(sorted user id of every review, reviews, popular counts) from `interactions`."""
    users, games, timestamps, positives = [], [], [], []
    columns = ["user_id", "game_idx", "timestamp", "is_positive"]
    for batch in source.batches("interactions", columns, snapshot_id):
        user = _int64(pc.fill_null(batch["user_id"], -1))
        game = _int64(pc.fill_null(batch["game_idx"], -1))
        keep = (user >= 0) & (game >= 0)
        users.append(user[keep])
        games.append(game[keep])
        timestamps.append(_micros(batch["timestamp"])[keep])
        positive = pc.fill_null(batch["is_positive"], False).to_numpy(zero_copy_only=False)
        positives.append(np.asarray(positive, dtype=bool)[keep])
    if not users:
        empty = np.zeros(0, dtype=np.int64)
        return empty, Reviews(empty), empty
    user = np.concatenate(users)
    game = np.concatenate(games)
    ts = np.concatenate(timestamps)
    popular = popular_counts(game, ts, np.concatenate(positives), popular_window_days)
    order = np.lexsort((-ts, user))
    return user[order], Reviews(game[order]), popular


def popular_counts(
    game_idx: np.ndarray, micros: np.ndarray, positive: np.ndarray, window_days: int
) -> np.ndarray:
    """Positive reviews per game_idx in the last `window_days` before the newest review."""
    if not len(game_idx):
        return np.zeros(0, dtype=np.int64)
    recent = positive & (micros >= micros.max() - window_days * 86_400 * 1_000_000)
    return np.bincount(game_idx[recent], minlength=int(game_idx.max()) + 1).astype(np.int64)


def _latest_users(batches: Iterable[pa.RecordBatch]) -> Users:
    """Latest row per user, reduced whenever the buffer outgrows the kept state."""
    state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    buffered = 0
    for batch in batches:
        user = _int64(pc.fill_null(batch["user_id"], -1))
        keep = user >= 0
        history = pad_history(Ragged.from_arrow(batch["games_reviewed_positive"]))
        buffer.append((user[keep], _micros(batch["timestamp"])[keep], history[keep]))
        buffered += int(keep.sum())
        if buffered > max(REDUCE_ROWS, 2 * (len(state[0]) if state else 0)):
            state = _reduce_latest([state, *buffer] if state else buffer)
            buffer, buffered = [], 0
    if buffer or state is None:
        state = _reduce_latest([state, *buffer] if state else buffer)
    user, _, history = state
    zeros = np.zeros(len(user), dtype=np.int64)
    return Users(user, history, zeros, zeros.copy())


def _reduce_latest(parts: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not parts:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, USER_HISTORY_LENGTH), dtype=np.int64),
        )
    user = np.concatenate([p[0] for p in parts])
    ts = np.concatenate([p[1] for p in parts])
    history = np.concatenate([p[2] for p in parts])
    order = np.lexsort((ts, user))
    ordered = user[order]
    is_last = np.ones(len(order), dtype=bool)
    is_last[:-1] = ordered[:-1] != ordered[1:]
    keep = order[is_last]
    return user[keep], ts[keep], history[keep]


def _lookup_names(source: TableSource, table: str, snapshot_id: int | None) -> np.ndarray:
    ids, names = [], []
    for batch in source.batches(table, ["id", "name"], snapshot_id):
        ids.append(_int64(pc.fill_null(batch["id"], -1)))
        names.extend(pc.fill_null(batch["name"], "").to_pylist())
    all_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    out = np.full(max(int(all_ids.max(initial=0)) + 1, 2), "", dtype=object)
    valid = all_ids >= 0
    out[all_ids[valid]] = np.asarray(names, dtype=object)[valid]
    return out


def _names(names: np.ndarray, ids: list[int]) -> list[str]:
    return [names[i] for i in ids if 0 <= i < len(names) and names[i]]


def _max_id(source: TableSource, table: str, column: str, snapshot_id: int | None) -> int:
    max_id = 1
    for batch in source.batches(table, [column], snapshot_id):
        if batch.num_rows:
            max_id = max(max_id, int(pc.max(batch[column]).as_py() or 0))
    return max_id


def _int64(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _micros(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    column = pc.fill_null(pc.cast(column, pa.timestamp("us")), 0)
    return pc.cast(column, pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)
