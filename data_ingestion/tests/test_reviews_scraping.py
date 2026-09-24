from __future__ import annotations

import io
from datetime import date
from types import SimpleNamespace

import polars as pl
import responses
from responses import matchers

from steam_ingestion.config import IngestionSettings
from steam_ingestion.models import PartitionFile
from steam_ingestion.reviews_scraping.scraper import build_review_record, scrape_partition
from steam_ingestion.schemas import REVIEWS_SCHEMA
from steam_ingestion.state import ReviewCursor, ReviewsCursorState
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


def _mock_reviews(appid: int, pages: list[list[dict]], total: int | None = None) -> None:
    """Newest-first pages; the last page echoes its own cursor back, as Steam does."""
    url = f"https://store.steampowered.com/appreviews/{appid}"
    for i, reviews in enumerate(pages):
        cursor_in = "*" if i == 0 else f"c{i}"
        last = i == len(pages) - 1
        body: dict = {
            "success": 1,
            "cursor": cursor_in if last else f"c{i + 1}",
            "reviews": reviews,
        }
        if i == 0:
            body["query_summary"] = {"total_reviews": total or sum(len(p) for p in pages)}
        responses.get(
            url,
            match=[matchers.query_param_matcher({"cursor": cursor_in}, strict_match=False)],
            json=body,
        )


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
