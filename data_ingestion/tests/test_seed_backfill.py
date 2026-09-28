from __future__ import annotations

from types import SimpleNamespace

from steam_ingestion.seed_backfill import Scraped, backfill_fields, query_scraped, seed
from steam_ingestion.state import ReviewCursor, ReviewsCursorState


def test_backfill_fields() -> None:
    capped = ReviewCursor(last_review_ts=900, total_reviews=10_000)
    assert backfill_fields(capped, Scraped(oldest=100, reviews=2000)) == (100, False)
    whole = ReviewCursor(last_review_ts=900, total_reviews=40)
    assert backfill_fields(whole, Scraped(oldest=100, reviews=40)) == (100, True)
    assert backfill_fields(ReviewCursor(0, 3), None) == (None, True)  # Steam returned nothing


def test_seed_writes_only_unseeded_cursors_and_is_idempotent(aws: SimpleNamespace) -> None:
    state = ReviewsCursorState(aws.cursors, aws.dynamodb)
    state.save(
        {
            1: ReviewCursor(900, 10_000),
            2: ReviewCursor(900, 40),
            3: ReviewCursor(900, 5, oldest_review_ts=50),  # written by the backfilling scraper
        }
    )
    scraped = {1: Scraped(100, 2000), 2: Scraped(200, 40), 3: Scraped(300, 5)}
    cursors = state.load_all()

    counts = seed(cursors, scraped, state.seed_backfill)
    assert counts == {"pending": 1, "complete": 1, "skipped": 1}
    loaded = state.load_all()
    assert loaded[1] == ReviewCursor(900, 10_000, oldest_review_ts=100, backfill_complete=False)
    assert loaded[2] == ReviewCursor(900, 40, oldest_review_ts=200, backfill_complete=True)
    assert loaded[3] == ReviewCursor(900, 5, oldest_review_ts=50)

    # stale snapshot: the conditional writes refuse to overwrite seeded cursors
    again = seed(cursors, scraped, state.seed_backfill, workers=4)
    assert again == {"skipped": 3}
    assert state.load_all() == loaded
    assert not state.seed_backfill(99, 1, False)  # no cursor: nothing created


class FakeAthena:
    """Two result pages; only the first one starts with the header row."""

    def __init__(self) -> None:
        self.states = iter(["RUNNING", "SUCCEEDED"])
        self.query = ""

    def start_query_execution(self, QueryString: str, WorkGroup: str) -> dict:
        self.query = QueryString
        return {"QueryExecutionId": "q1"}

    def get_query_execution(self, QueryExecutionId: str) -> dict:
        return {"QueryExecution": {"Status": {"State": next(self.states)}}}

    def get_paginator(self, name: str) -> SimpleNamespace:
        def row(*values: str) -> dict:
            return {"Data": [{"VarCharValue": v} for v in values]}

        pages = [
            {"ResultSet": {"Rows": [row("appid", "oldest", "reviews"), row("10", "100", "5")]}},
            {"ResultSet": {"Rows": [row("20", "200", "7")]}},
        ]
        return SimpleNamespace(paginate=lambda **_: pages)


def test_query_scraped_parses_all_pages(monkeypatch) -> None:
    monkeypatch.setattr("steam_ingestion.seed_backfill.time.sleep", lambda _: None)
    athena = FakeAthena()
    assert query_scraped(athena, "wg", "steam_raw") == {
        10: Scraped(100, 5),
        20: Scraped(200, 7),
    }
    assert "from steam_raw.reviews" in athena.query
