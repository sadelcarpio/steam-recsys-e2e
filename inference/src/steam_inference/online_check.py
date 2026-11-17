"""Delayed-label check (spec 9): the batch path's live metric, without A/B infrastructure.

Each run saves the lists of a sample of users (`ONLINE_CHECK_USERS`): the order it wrote and
the alternative order (shadow mode: the ranker's lists it did not write; ranker on: retrieval
order; no ranker: the same lists). The next run scores that sample against what the users
reviewed positively in between (games they had not reviewed before the earlier run), with
precision / recall / MAP@k of `steam_training.ranker.lists` (the per-user list metrics the
ranker report uses), and writes `evaluation/online/<run>.json`.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from botocore.exceptions import ClientError
from steam_training.data import Ragged
from steam_training.ranker.lists import ListAccumulator

from steam_inference.contracts import OnlineCheck
from steam_inference.features import InferenceData
from steam_inference.retrieval import NO_ROW, Candidates

log = logging.getLogger(__name__)
LATEST = "lists/latest.npz"
KS = (10, 20)


@dataclass(frozen=True)
class ListsSample:
    run_at: datetime  # UTC
    mode: str
    ranker_id: str | None
    user_id: np.ndarray  # int64 [n], sorted
    written: np.ndarray  # int64 [n, k] game_idx, best first, -1 = none
    alternative: np.ndarray  # int64 [n, k']

    def encode(self) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            run_at=np.int64(_micros(self.run_at)),
            mode=np.str_(self.mode),
            ranker_id=np.str_(self.ranker_id or ""),
            user_id=self.user_id,
            written=self.written,
            alternative=self.alternative,
        )
        return buffer.getvalue()

    @classmethod
    def decode(cls, body: bytes) -> ListsSample:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            return cls(
                run_at=datetime.fromtimestamp(int(z["run_at"]) / 1e6, UTC),
                mode=str(z["mode"]),
                ranker_id=str(z["ranker_id"]) or None,
                user_id=z["user_id"],
                written=z["written"],
                alternative=z["alternative"],
            )


def sample_lists(
    data: InferenceData,
    written: Candidates,
    alternative: Candidates,
    *,
    users: int,
    run_at: datetime,
    mode: str,
    ranker_id: str | None,
    seed: int = 0,
) -> ListsSample:
    """A random sample of the users with a list (`users` at most), as game_idx."""
    eligible = np.flatnonzero(written.rows[:, 0] != NO_ROW)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(eligible, min(users, len(eligible)), replace=False))
    game_idx = data.games.catalog.items.game_idx

    def games(rows: np.ndarray) -> np.ndarray:
        return np.where(rows != NO_ROW, game_idx[np.maximum(rows, 0)], -1).astype(np.int64)

    return ListsSample(
        run_at=run_at,
        mode=mode,
        ranker_id=ranker_id,
        user_id=data.users.user_id[pick],
        written=games(written.rows[pick]),
        alternative=games(alternative.rows[pick]),
    )


def check(previous: ListsSample, data: InferenceData, run_at: datetime) -> OnlineCheck | None:
    """The previous run's sample against the reviews written since (None: no user of the
    sample reviewed anything new, or the reviews carry no timestamps)."""
    reviews, users = data.reviews, data.users
    if reviews.micros is None or reviews.is_positive is None:
        return None
    since, until = _micros(previous.run_at), _micros(run_at)
    position = np.searchsorted(users.user_id, previous.user_id)
    position[position >= len(users)] = 0
    found = users.user_id[position] == previous.user_id if len(users) else position < 0
    keep, relevant = [], []
    for i in np.flatnonzero(found):
        start, count = users.review_start[position[i]], users.review_count[position[i]]
        span = slice(start, start + count)
        micros, game, positive = reviews.micros[span], reviews.game_idx[span], reviews.is_positive[
            span
        ]  # fmt: skip
        seen = game[micros <= since]
        new = game[positive & (micros > since) & (micros <= until)]
        new = np.unique(new[~np.isin(new, seen)])
        if len(new):
            keep.append(i)
            relevant.append(new.tolist())
    if not keep:
        return None
    keep = np.array(keep)
    targets = Ragged.from_lists(relevant)
    out = []
    for lists in (previous.written, previous.alternative):
        accumulator = ListAccumulator(list(KS))  # a shorter list scores 0 past its end
        accumulator.add(lists[keep], targets)
        out.append(accumulator.metrics())
    result = OnlineCheck(
        previous_run_at=previous.run_at,
        run_at=run_at,
        previous_mode=previous.mode,
        previous_ranker_id=previous.ranker_id,
        sampled_users=len(previous.user_id),
        written=out[0],
        alternative=out[1],
    )
    log.info(
        "online check: %d of %d sampled users reviewed since %s; written %s, alternative %s",
        len(keep),
        len(previous.user_id),
        previous.run_at,
        _short(result.written),
        _short(result.alternative),
    )
    return result


def load_sample(s3, bucket: str, prefix: str) -> ListsSample | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=f"{prefix}/{LATEST}")["Body"].read()
    except ClientError as err:
        if err.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return ListsSample.decode(body)


def save(s3, bucket: str, prefix: str, sample: ListsSample, result: OnlineCheck | None) -> None:
    """The report first (evaluation/online/<run>.json), then the new sample (the next run's
    input)."""
    if result is not None:
        key = f"{prefix}/{result.run_at:%Y%m%dT%H%M%SZ}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=result.model_dump_json(indent=2).encode(),
            ContentType="application/json",
        )
        log.info("wrote s3://%s/%s", bucket, key)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{LATEST}",
        Body=sample.encode(),
        ContentType="application/octet-stream",
    )


def _micros(value: datetime) -> int:
    value = value if value.tzinfo else value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000)


def _short(metrics) -> str:
    return " ".join(f"map@{k}={v:.4f}" for k, v in metrics.map.items())
