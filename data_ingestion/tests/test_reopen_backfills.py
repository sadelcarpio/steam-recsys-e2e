from __future__ import annotations

from types import SimpleNamespace

from steam_ingestion.reopen_backfills import reopen, to_reopen
from steam_ingestion.seed_backfill import Scraped
from steam_ingestion.state import ReviewCursor, ReviewsCursorState


def _complete(total: int, stored: int | None = None) -> ReviewCursor:
    return ReviewCursor(
        900, total, oldest_review_ts=100, backfill_complete=True, stored_reviews=stored
    )


def test_to_reopen() -> None:
    few = Scraped(oldest=100, reviews=2_000)
    assert to_reopen(_complete(1_400_000), few, 500_000) == 2_000  # ended early
    assert to_reopen(_complete(2_050), few, 500_000) is None  # within the slack
    assert to_reopen(_complete(10_000_000, stored=500_000), few, 500_000) is None  # capped
    many = Scraped(oldest=100, reviews=600_000)
    assert to_reopen(_complete(10_000_000), many, 500_000) is None  # capped, count from raw
    assert to_reopen(_complete(10_000_000), many, 0) == 600_000  # no cap
    pending = ReviewCursor(900, 1_400_000, oldest_review_ts=100)
    assert to_reopen(pending, few, 500_000) is None
    no_oldest = ReviewCursor(900, 1_400_000, backfill_complete=True)  # nothing to walk from
    assert to_reopen(no_oldest, few, 500_000) is None
    assert to_reopen(_complete(1_000), None, 500_000) == 0  # nothing in raw


def test_reopen_marks_pending_seeds_the_count_and_is_idempotent(aws: SimpleNamespace) -> None:
    state = ReviewsCursorState(aws.cursors, aws.dynamodb)
    state.save(
        {
            1: _complete(1_400_000),  # ended early, never counted
            2: _complete(900_000, stored=3_000),  # ended early, counted: the count is kept
            3: _complete(2_050),  # really complete
            4: _complete(10_000_000, stored=500_000),  # at the cap
            5: ReviewCursor(900, 1_400_000, oldest_review_ts=100),  # already pending
        }
    )
    scraped = {a: Scraped(oldest=100, reviews=2_000) for a in range(1, 6)}
    cursors = state.load_all()

    counts = reopen(cursors, scraped, state.reopen_backfill, max_stored=500_000)
    assert counts == {"reopened": 2, "skipped": 3, "missing": 1_398_000 + 898_000}
    loaded = state.load_all()
    assert [a for a, c in loaded.items() if c.backfill_pending] == [1, 2, 5]
    assert loaded[1].stored_reviews == 2_000 and loaded[2].stored_reviews == 3_000
    assert loaded[3] == cursors[3] and loaded[4] == cursors[4]

    # stale snapshot: the conditional writes refuse to reopen twice
    assert reopen(cursors, scraped, state.reopen_backfill, max_stored=500_000) == {"skipped": 5}
    assert state.load_all() == loaded
