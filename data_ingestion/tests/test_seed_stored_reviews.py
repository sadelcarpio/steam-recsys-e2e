from __future__ import annotations

from types import SimpleNamespace

from steam_ingestion.seed_backfill import Scraped
from steam_ingestion.seed_stored_reviews import seed
from steam_ingestion.state import ReviewCursor, ReviewsCursorState


def test_seed_counts_only_pending_cursors_and_is_idempotent(aws: SimpleNamespace) -> None:
    state = ReviewsCursorState(aws.cursors, aws.dynamodb)
    state.save(
        {
            1: ReviewCursor(900, 10_000, oldest_review_ts=100),  # pending, over the cap
            2: ReviewCursor(900, 10_000, oldest_review_ts=100),  # pending, under the cap
            3: ReviewCursor(900, 40, oldest_review_ts=100, backfill_complete=True),
            4: ReviewCursor(900, 10_000, oldest_review_ts=100, stored_reviews=9),  # counted
            5: ReviewCursor(900, 10_000, oldest_review_ts=100),  # pending, nothing in raw
        }
    )
    scraped = {a: Scraped(oldest=100, reviews=r) for a, r in {1: 600, 2: 300, 3: 40, 4: 8}.items()}
    cursors = state.load_all()

    counts = seed(cursors, scraped, state.seed_stored_reviews, max_stored=500)
    assert counts == {"capped": 1, "pending": 2, "skipped": 2}
    loaded = state.load_all()
    assert {a: c.stored_reviews for a, c in loaded.items()} == {1: 600, 2: 300, 3: None, 4: 9, 5: 0}

    # stale snapshot: the conditional writes refuse to overwrite seeded counts
    assert seed(cursors, scraped, state.seed_stored_reviews, max_stored=500) == {"skipped": 5}
    assert state.load_all() == loaded
