from __future__ import annotations

import io
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
import responses
from responses import matchers

from steam_ingestion.config import IngestionSettings
from steam_ingestion.models import PartitionFile
from steam_ingestion.reviews_scraping.scraper import build_review_record, scrape_partition
from steam_ingestion.schemas import REVIEWS_SCHEMA
from steam_ingestion.shutdown import Shutdown, ShutdownRequested
from steam_ingestion.state import ReviewCursor, ReviewsCursorState
from steam_ingestion.steam_api import SteamClient
from steam_ingestion.storage import list_keys, write_partition

from .conftest import PARTITIONS_BUCKET, RAW_BUCKET

TODAY = date(2026, 9, 24)
KEY = "reviews/run-1/part-002.json"
PREFIX = "reviews/2026-09-24-002-"


def _review(rec_id: int, ts: int) -> dict:
    return {
        "recommendationid": str(rec_id),
        "author": {
            "steamid": "76561197960287930",
            "num_games_owned": 10,
            "num_reviews": 2,
            "playtime_forever": 100,
            "playtime_last_two_weeks": 0,
            "playtime_at_review": 50,
            "last_played": 1700000000,
        },
        "language": "english",
        "review": f"review {rec_id}",
        "timestamp_created": ts,
        "timestamp_updated": ts,
        "voted_up": True,
        "votes_up": 1,
        "votes_funny": 0,
        "weighted_vote_score": "0.523809552192687988",
        "comment_count": 0,
        "steam_purchase": True,
        "received_for_free": False,
        "written_during_early_access": False,
        "primarily_steam_deck": False,
    }


def _no_range(request) -> tuple[bool, str]:
    return "end_date" not in request.params, "forward requests carry no date range"


def _mock_reviews(
    appid: int, pages: list[list[dict]], total: int | None = None, until: int | None = None
) -> None:
    """Newest-first pages; the last page echoes its own cursor back, as Steam does. `until`:
    the backfill range `[1, until]` (else only requests without a date range match)."""
    url = f"https://store.steampowered.com/appreviews/{appid}"
    prefix = "" if until is None else f"b{until}-"
    for i, reviews in enumerate(pages):
        cursor_in = "*" if i == 0 else f"{prefix}c{i}"
        last = i == len(pages) - 1
        body: dict = {
            "success": 1,
            "cursor": cursor_in if last else f"{prefix}c{i + 1}",
            "reviews": reviews,
        }
        if i == 0:
            body["query_summary"] = {"total_reviews": total or sum(len(p) for p in pages)}
        params = {"cursor": cursor_in}
        if until is not None:
            params |= {"start_date": "1", "end_date": str(until)}
        match = [matchers.query_param_matcher(params, strict_match=False)]
        responses.get(url, match=match if until is not None else [*match, _no_range], json=body)


def _seed(aws: SimpleNamespace, appids: list[int]) -> None:
    write_partition(aws.s3, PARTITIONS_BUCKET, KEY, PartitionFile(run_id="run-1", appids=appids))


def _rows(aws: SimpleNamespace) -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_parquet(io.BytesIO(aws.s3.get_object(Bucket=RAW_BUCKET, Key=k)["Body"].read()))
            for k in sorted(list_keys(aws.s3, RAW_BUCKET, PREFIX))
        ]
    )


def _cursor(aws: SimpleNamespace, appid: int) -> dict | None:
    return aws.cursors.get_item(Key={"appid": appid}).get("Item")


def test_build_review_record_matches_schema() -> None:
    record = build_review_record(10, _review(1, 100), TODAY)
    assert record.author_id == 76561197960287930
    assert record.weighted_vote_score == 0.523809552192687988
    df = pl.DataFrame([record.model_dump()], schema=REVIEWS_SCHEMA)
    assert df.schema == REVIEWS_SCHEMA


@responses.activate
def test_first_run_scrapes_all_and_saves_cursors(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10, 20])
    _mock_reviews(10, [[_review(3, 300), _review(2, 200)], [_review(1, 100)]])
    _mock_reviews(20, [[_review(4, 400)]])

    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    df = _rows(aws)
    assert df.schema == REVIEWS_SCHEMA
    assert sorted(df["rec_id"].to_list()) == [1, 2, 3, 4]
    assert int(_cursor(aws, 10)["last_review_ts"]) == 300
    assert int(_cursor(aws, 10)["total_reviews"]) == 3
    assert int(_cursor(aws, 20)["last_review_ts"]) == 400


@responses.activate
def test_incremental_run_only_fetches_new_reviews(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=200, total_reviews=2)}
    )
    _mock_reviews(10, [[_review(3, 300), _review(2, 200), _review(1, 100)]])

    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    assert _rows(aws)["rec_id"].to_list() == [3]
    assert int(_cursor(aws, 10)["last_review_ts"]) == 300
    assert len(responses.calls) == 1  # stopped at the cutoff, no second page


@responses.activate
def test_no_new_reviews_keeps_cursor_and_writes_nothing(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=300, total_reviews=1)}
    )
    _mock_reviews(10, [[_review(3, 300)]])
    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert list_keys(aws.s3, RAW_BUCKET, "reviews/") == []
    assert int(_cursor(aws, 10)["last_review_ts"]) == 300


@responses.activate
def test_mid_game_flush_does_not_advance_cursor_of_unfinished_game(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    """reviews_flush_rows=3: game 20's first page triggers a flush, then its 2nd page fails."""
    _seed(aws, [10, 20])
    _mock_reviews(10, [[_review(1, 100)]])
    url = "https://store.steampowered.com/appreviews/20"
    responses.get(
        url,
        match=[matchers.query_param_matcher({"cursor": "*"}, strict_match=False)],
        json={
            "success": 1,
            "cursor": "c1",
            "query_summary": {"total_reviews": 5},
            "reviews": [_review(5, 500), _review(4, 400)],
        },
    )
    responses.get(
        url, match=[matchers.query_param_matcher({"cursor": "c1"}, strict_match=False)], status=503
    )

    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    assert sorted(_rows(aws)["rec_id"].to_list()) == [1, 4, 5]  # flushed mid-game
    assert int(_cursor(aws, 10)["last_review_ts"]) == 100  # finished before the flush
    assert _cursor(aws, 20) is None  # unfinished: re-scraped next run


@responses.activate
def test_retry_continues_part_numbering(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    aws.s3.put_object(Bucket=RAW_BUCKET, Key=f"{PREFIX}0000.parquet", Body=b"previous attempt")
    _mock_reviews(10, [[_review(1, 100)]])
    scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert sorted(list_keys(aws.s3, RAW_BUCKET, PREFIX)) == [
        f"{PREFIX}0000.parquet",
        f"{PREFIX}0001.parquet",
    ]
    body = aws.s3.get_object(Bucket=RAW_BUCKET, Key=f"{PREFIX}0000.parquet")["Body"].read()
    assert body == b"previous attempt"


@responses.activate
def test_cursor_load_batches_over_100_keys(aws: SimpleNamespace) -> None:
    state = ReviewsCursorState(aws.cursors, aws.dynamodb)
    state.save({a: ReviewCursor(last_review_ts=a, total_reviews=None) for a in range(1, 251)})
    loaded = state.load(list(range(1, 301)))
    assert len(loaded) == 250 and loaded[250].last_review_ts == 250


def _backfill_state(cursor: dict) -> tuple[int | None, bool]:
    oldest = cursor.get("oldest_review_ts")
    return (int(oldest) if oldest is not None else None), bool(cursor.get("backfill_complete"))


@responses.activate
def test_first_scrape_within_the_cap_is_complete(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    _mock_reviews(10, [[_review(2, 200), _review(1, 100)]])
    capped = settings.model_copy(update={"max_reviews_per_game": 5})
    assert scrape_partition(KEY, capped, client, aws.s3, aws.dynamodb, today=TODAY)
    assert _backfill_state(_cursor(aws, 10)) == (100, True)
    assert len(responses.calls) == 1  # no backfill request


@responses.activate
def test_first_scrape_truncated_by_the_cap_backfills_in_the_same_run(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    _mock_reviews(10, [[_review(5, 500), _review(4, 400)], [_review(3, 300)]], total=5)
    # older than 400 (inclusive): the boundary review repeats, then the rest of the history
    _mock_reviews(10, [[_review(4, 400), _review(3, 300)], [_review(2, 200)]], until=400)
    run = settings.model_copy(update={"max_reviews_per_game": 2, "backfill_reviews_per_run": 3})

    assert scrape_partition(KEY, run, client, aws.s3, aws.dynamodb, today=TODAY)

    assert sorted(_rows(aws)["rec_id"].to_list()) == [2, 3, 4, 4, 5]
    cursor = _cursor(aws, 10)
    assert int(cursor["last_review_ts"]) == 500 and int(cursor["total_reviews"]) == 5
    assert _backfill_state(cursor) == (200, False)  # budget of 3 reached: more may remain


@responses.activate
def test_pending_backfill_continues_and_completes(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=500, total_reviews=6, oldest_review_ts=300)}
    )
    _mock_reviews(10, [[_review(6, 600), _review(5, 500)]])  # forward: one new review
    _mock_reviews(10, [[_review(3, 300), _review(2, 200)], [_review(1, 100)]], until=300)

    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    assert sorted(_rows(aws)["rec_id"].to_list()) == [1, 2, 3, 6]
    cursor = _cursor(aws, 10)
    assert int(cursor["last_review_ts"]) == 600
    assert _backfill_state(cursor) == (100, True)  # the range ran out: history complete


@responses.activate
def test_complete_and_legacy_cursors_are_not_backfilled(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10, 20])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {
            10: ReviewCursor(300, 3, oldest_review_ts=100, backfill_complete=True),
            20: ReviewCursor(300, 3),  # written before the backfill and not seeded
        }
    )
    _mock_reviews(10, [[_review(3, 300)]])
    _mock_reviews(20, [[_review(13, 300)]])
    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert len(responses.calls) == 2  # forward passes only
    assert _backfill_state(_cursor(aws, 10)) == (100, True)
    assert _backfill_state(_cursor(aws, 20)) == (None, False)


@responses.activate
def test_failed_backfill_keeps_forward_progress_and_what_was_fetched(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=500, total_reviews=9, oldest_review_ts=300)}
    )
    _mock_reviews(10, [[_review(6, 600)]])
    url = "https://store.steampowered.com/appreviews/10"
    responses.get(
        url,
        match=[
            matchers.query_param_matcher({"cursor": "*", "end_date": "300"}, strict_match=False)
        ],
        json={"success": 1, "cursor": "b1", "reviews": [_review(2, 200)]},
    )
    responses.get(
        url,
        match=[matchers.query_param_matcher({"cursor": "b1"}, strict_match=False)],
        status=503,
    )

    # a failed backfill does not count as a failed game
    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)

    assert sorted(_rows(aws)["rec_id"].to_list()) == [2, 6]
    cursor = _cursor(aws, 10)
    assert int(cursor["last_review_ts"]) == 600
    assert _backfill_state(cursor) == (200, False)


@responses.activate
def test_ignored_date_range_does_not_complete_the_backfill(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=500, total_reviews=9, oldest_review_ts=300)}
    )
    _mock_reviews(10, [[_review(5, 500)]])
    _mock_reviews(10, [[_review(5, 500)]], until=300)  # Steam returned the newest instead
    assert scrape_partition(KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY)
    assert list_keys(aws.s3, RAW_BUCKET, "reviews/") == []
    assert _backfill_state(_cursor(aws, 10)) == (300, False)


@responses.activate
def test_backfill_disabled_with_a_zero_budget(
    aws: SimpleNamespace, settings: IngestionSettings, client
) -> None:
    _seed(aws, [10])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=500, total_reviews=9, oldest_review_ts=300)}
    )
    _mock_reviews(10, [[_review(5, 500)]])
    off = settings.model_copy(update={"backfill_reviews_per_run": 0})
    assert scrape_partition(KEY, off, client, aws.s3, aws.dynamodb, today=TODAY)
    assert len(responses.calls) == 1


def _stop_after_pages(client: SteamClient, shutdown: Shutdown, n: int) -> None:
    """Request a stop (SIGTERM) while the `n`-th review page overall is being processed."""
    real, seen = client.iter_review_pages, 0

    def pages(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal seen
        for page in real(*args, **kwargs):
            seen += 1
            if seen == n:
                shutdown.request()
            yield page

    client.iter_review_pages = pages  # type: ignore[method-assign]


@responses.activate
def test_stop_during_a_forward_pass_flushes_finished_games_only(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, [10, 20, 30])
    _mock_reviews(10, [[_review(1, 100)]])
    _mock_reviews(20, [[_review(3, 300)], [_review(2, 200)]])
    shutdown = Shutdown()
    _stop_after_pages(client, shutdown, 2)  # game 20's first page

    with pytest.raises(ShutdownRequested):
        scrape_partition(
            KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY, shutdown=shutdown
        )

    assert sorted(_rows(aws)["rec_id"].to_list()) == [1, 3]  # the buffer is written
    assert int(_cursor(aws, 10)["last_review_ts"]) == 100
    assert _cursor(aws, 20) is None  # forward walk unfinished: re-scraped by the retry
    assert _cursor(aws, 30) is None
    assert len(responses.calls) == 2


@responses.activate
def test_stop_during_a_backfill_commits_what_was_fetched(
    aws: SimpleNamespace, settings: IngestionSettings, client: SteamClient
) -> None:
    _seed(aws, [10, 20])
    ReviewsCursorState(aws.cursors, aws.dynamodb).save(
        {10: ReviewCursor(last_review_ts=500, total_reviews=9, oldest_review_ts=300)}
    )
    _mock_reviews(10, [[_review(6, 600)]])
    _mock_reviews(10, [[_review(3, 300), _review(2, 200)], [_review(1, 100)]], until=300)
    shutdown = Shutdown()
    _stop_after_pages(client, shutdown, 2)  # the backfill's first page

    with pytest.raises(ShutdownRequested):
        scrape_partition(
            KEY, settings, client, aws.s3, aws.dynamodb, today=TODAY, shutdown=shutdown
        )

    assert sorted(_rows(aws)["rec_id"].to_list()) == [2, 3, 6]
    cursor = _cursor(aws, 10)
    assert int(cursor["last_review_ts"]) == 600
    assert _backfill_state(cursor) == (200, False)  # moved back as far as it got
    assert _cursor(aws, 20) is None


def test_cursor_backfill_fields_roundtrip(aws: SimpleNamespace) -> None:
    state = ReviewsCursorState(aws.cursors, aws.dynamodb)
    cursors = {
        1: ReviewCursor(10, 5, oldest_review_ts=3, backfill_complete=False),
        2: ReviewCursor(10, 5, oldest_review_ts=None, backfill_complete=True),
        3: ReviewCursor(10, None),
    }
    state.save(cursors)
    assert state.load([1, 2, 3]) == cursors
    assert state.load_all() == cursors
    assert [c.backfill_pending for c in cursors.values()] == [True, False, False]
