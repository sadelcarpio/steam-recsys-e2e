"""Ranker data (spec 9): query groups after a retriever's cutoff, and the game-side tables.

One group per kept positive review (u, g, t) at or after the retriever's cutoff T1 whose user has
a positive review before t (warm: inference never ranks an empty history). The rows come from
`data.load_training_data` with that cutoff (the same validation loader promotion uses: a uniform
sample, each row's history and the games reviewed before it), then:
  - at most `max_groups_per_user` groups per user (heavy reviewers don't dominate);
  - one more streamed pass over `interactions`: weekly positive / total reviews of every game
    (`PopularityCounter`) and each kept user's positive reviews in the window, so a group can
    drop games its user reviews positively later (they are not negatives);
  - prices and release dates from `game_details` (via `lkp_games`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pyarrow.compute as pc

from steam_training.contracts import VocabSizes
from steam_training.data import (
    Catalog,
    Ragged,
    TableSource,
    _bool,
    _int64,
    _seconds,
    _timestamps,
    load_training_data,
)
from steam_training.ranker.features import (
    MAX_RECENT_WEEKS,
    SECONDS_PER_DAY,
    GameTable,
    PopularityCounter,
    PopularityTable,
    week_of,
)

log = logging.getLogger(__name__)

RELEASE_DATE_FORMATS = ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y")


@dataclass(frozen=True)
class RankerRows:
    """Query groups, one per row."""

    user_id: np.ndarray  # int64
    seconds: np.ndarray  # int64, time t
    history: np.ndarray  # int64 [n, length], positives before t, most recent first, 0 = pad
    target: np.ndarray  # int64 game_idx
    reviewed: Ragged  # every game reviewed before t (excluded from the candidates)
    later: Ragged  # the user's other positives after t (dropped from the candidates)

    def __len__(self) -> int:
        return len(self.target)

    def take(self, index: np.ndarray) -> RankerRows:
        return RankerRows(
            self.user_id[index],
            self.seconds[index],
            self.history[index],
            self.target[index],
            self.reviewed.take(index),
            self.later.take(index),
        )


@dataclass(frozen=True)
class RankerData:
    rows: RankerRows  # sorted by time
    catalog: Catalog  # as of the cutoff (the retriever's corpus)
    vocab: VocabSizes
    games: GameTable
    popularity: PopularityTable
    snapshots: dict[str, int | None]
    cutoff: datetime


def load_ranker_data(
    source: TableSource,
    *,
    cutoff: datetime,
    history_length: int,
    max_rows: int,
    max_groups_per_user: int,
    seed: int,
) -> RankerData:
    data = load_training_data(
        source,
        validation_fraction=0.1,  # unused: the cutoff is given
        cold_row_fraction=0.0,
        eval_max_rows=max_rows,
        seed=seed,
        cutoff=cutoff,
        with_train_rows=False,
        with_tags=True,
        history_length=history_length,
    )
    val = data.validation
    if val.user_id is None or val.timestamp is None or val.reviewed is None:
        raise ValueError("validation rows without users / timestamps")
    warm = np.flatnonzero(val.is_warm)
    kept = warm[cap_per_user(val.user_id[warm], max_groups_per_user, seed)]
    kept = kept[np.argsort(val.timestamp[kept], kind="stable")]
    log.info(
        "ranker groups: %d warm of %d sampled rows, %d after the per-user cap of %d",
        len(warm),
        len(val),
        len(kept),
        max_groups_per_user,
    )
    user_id, seconds, target = val.user_id[kept], val.timestamp[kept], val.target[kept]
    cutoff_seconds = epoch_seconds(cutoff)
    end_seconds = int(seconds.max()) if len(kept) else cutoff_seconds
    first_week = int(week_of(cutoff_seconds)) - MAX_RECENT_WEEKS
    counter = PopularityCounter(data.vocab.games, first_week, int(week_of(end_seconds)))
    later = _window_pass(
        source, data.snapshots["interactions"], counter, user_id, seconds, target, cutoff
    )
    snapshots = dict(data.snapshots)
    for table in ("lkp_games", "game_details"):
        snapshots[table] = source.snapshot_id(table)
    price, release_day = game_details(source, snapshots, data.vocab.games)
    games = GameTable.from_catalog(
        data.catalog,
        data.vocab.games,
        data.vocab.tags or 0,
        data.vocab.genres,
        data.vocab.categories,
        price=price,
        release_day=release_day,
    )
    rows = RankerRows(
        user_id=user_id,
        seconds=seconds,
        history=val.history[kept],
        target=target,
        reviewed=val.reviewed.take(kept),
        later=later,
    )
    return RankerData(
        rows=rows,
        catalog=data.catalog,
        vocab=data.vocab,
        games=games,
        popularity=counter.table(),
        snapshots=snapshots,
        cutoff=cutoff,
    )


def epoch_seconds(value: datetime) -> int:
    """Seconds since the epoch of a naive UTC datetime (the marts' and `SplitInfo`'s), without
    the local-timezone shift of `datetime.timestamp()`."""
    return int(np.datetime64(value.replace(tzinfo=None), "s").astype(np.int64))


def from_epoch_seconds(seconds: int) -> datetime:
    """Naive UTC datetime of `seconds` since the epoch."""
    return np.datetime64(int(seconds), "s").astype("datetime64[us]").item()


def cap_per_user(user_id: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Indices of at most `cap` random rows per user (sorted)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(user_id))
    shuffled = user_id[order]
    # rank of each row within its user, in the shuffled order
    by_user = np.argsort(shuffled, kind="stable")
    sorted_users = shuffled[by_user]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    within = np.arange(len(sorted_users)) - np.repeat(
        starts, np.diff(np.r_[starts, len(sorted_users)])
    )
    keep = np.zeros(len(user_id), dtype=bool)
    keep[order[by_user[within < cap]]] = True
    return np.flatnonzero(keep)


def _window_pass(
    source: TableSource,
    snapshot: int | None,
    counter: PopularityCounter,
    row_user: np.ndarray,
    row_seconds: np.ndarray,
    row_target: np.ndarray,
    cutoff: datetime,
) -> Ragged:
    """Fills `counter` from every review and returns, per row, the positives of its user after
    t (at or after the cutoff), its target left out."""
    users = np.unique(row_user)
    cutoff_us = np.datetime64(cutoff, "us")
    user_parts, time_parts, game_parts = [], [], []
    for batch in source.batches(
        "interactions", ["timestamp", "user_id", "game_idx", "is_positive"], snapshot
    ):
        ts = _timestamps(batch["timestamp"])
        seconds = _seconds(ts)
        game = _int64(pc.fill_null(batch["game_idx"], -1))
        positive = _bool(batch["is_positive"])
        counter.add(game, seconds, positive)
        ids = _int64(pc.fill_null(batch["user_id"], -1))
        position = np.searchsorted(users, ids)
        position[position >= len(users)] = 0
        found = (users[position] == ids) if len(users) else np.zeros(len(ids), dtype=bool)
        keep = found & positive & (ts >= cutoff_us) & (game >= 0)
        user_parts.append(position[keep])
        time_parts.append(seconds[keep])
        game_parts.append(game[keep])
    return positives_after(
        np.concatenate(user_parts) if user_parts else np.zeros(0, np.int64),
        np.concatenate(time_parts) if time_parts else np.zeros(0, np.int64),
        np.concatenate(game_parts) if game_parts else np.zeros(0, np.int64),
        np.searchsorted(users, row_user),
        row_seconds,
        row_target,
    )


def positives_after(
    user: np.ndarray,
    seconds: np.ndarray,
    game: np.ndarray,
    row_user: np.ndarray,
    row_seconds: np.ndarray,
    row_target: np.ndarray,
) -> Ragged:
    """Per row, the games of `row_user` with seconds > `row_seconds`, minus the row's target.
    Users are dense ids (as in `data.reviewed_before`)."""
    key = (user.astype(np.int64) << 32) | seconds.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, game = key[order], game[order]
    base = row_user.astype(np.int64) << 32
    lo = np.searchsorted(key, base | row_seconds.astype(np.int64), side="right")
    hi = np.searchsorted(key, base | np.int64(0xFFFFFFFF), side="right")
    lengths = np.maximum(hi - lo, 0)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    within = np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], lengths)
    values = game[np.repeat(lo, lengths) + within]
    keep = values != np.repeat(row_target, lengths)
    kept = np.bincount(np.repeat(np.arange(len(lengths)), lengths)[keep], minlength=len(lengths))
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(kept, out=offsets[1:])
    return Ragged(values[keep], offsets)


def game_details(
    source: TableSource, snapshots: dict[str, int | None], num_games: int
) -> tuple[np.ndarray, np.ndarray]:
    """(price, release day) per game_idx from `game_details` + `lkp_games`, NaN when unknown.
    The price is the one at the last scrape (not as of t)."""
    index: dict[int, int] = {}
    for batch in source.batches("lkp_games", ["game_id", "game_idx"], snapshots["lkp_games"]):
        ids, idx = _int64(batch["game_id"]).tolist(), _int64(batch["game_idx"]).tolist()
        index.update(zip(ids, idx, strict=True))
    price = np.full(num_games, np.nan, dtype=np.float32)
    release = np.full(num_games, np.nan, dtype=np.float32)
    parsed: dict[str, float] = {}
    columns = ["game_id", "game_price", "game_release_date"]
    for batch in source.batches("game_details", columns, snapshots["game_details"]):
        ids = _int64(batch["game_id"]).tolist()
        prices = batch["game_price"].to_pylist()
        dates = batch["game_release_date"].to_pylist()
        for game_id, value, text in zip(ids, prices, dates, strict=True):
            idx = index.get(game_id)
            if idx is None or not 0 <= idx < num_games:
                continue
            if value is not None:
                price[idx] = value
            if text:
                if text not in parsed:
                    parsed[text] = parse_release_day(text)
                release[idx] = parsed[text]
    return price, release


def parse_release_day(text: str) -> float:
    """Days since the epoch of a Steam release date ("Oct 9, 2020", "9 Oct, 2020"); NaN for
    month-only, "Coming soon", etc."""
    text = text.strip()
    for fmt in RELEASE_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() / SECONDS_PER_DAY
        except ValueError:
            continue
    return float("nan")
