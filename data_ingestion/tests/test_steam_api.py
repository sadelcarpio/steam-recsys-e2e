from __future__ import annotations

import pytest
import requests
import responses
from responses import matchers

from steam_ingestion.steam_api import (
    APP_DETAILS_URL,
    APP_LIST_URL,
    SteamApiError,
    SteamClient,
)

REVIEWS_URL = "https://store.steampowered.com/appreviews/10"


def _review(rec_id: int, ts: int) -> dict:
    return {"recommendationid": str(rec_id), "timestamp_created": ts, "author": {"steamid": "1"}}


@responses.activate
def test_get_app_list_paginates_and_sends_modified_since(client: SteamClient) -> None:
    responses.get(
        APP_LIST_URL,
        match=[
            matchers.query_param_matcher(
                {"last_appid": "0", "if_modified_since": "123"}, strict_match=False
            )
        ],
        json={
            "response": {
                "apps": [{"appid": 10, "last_modified": 200}],
                "have_more_results": True,
                "last_appid": 10,
            }
        },
    )
    responses.get(
        APP_LIST_URL,
        match=[matchers.query_param_matcher({"last_appid": "10"}, strict_match=False)],
        json={"response": {"apps": [{"appid": 20, "last_modified": 300}]}},
    )
    apps = client.get_app_list(if_modified_since=123)
    assert [(a.appid, a.last_modified) for a in apps] == [(10, 200), (20, 300)]
    assert "include_dlc=false" in responses.calls[0].request.url


@responses.activate
def test_get_app_list_handles_empty_response(client: SteamClient) -> None:
    responses.get(APP_LIST_URL, json={"response": {}})
    assert client.get_app_list() == []


@responses.activate
def test_retries_on_throttling_and_null_body(client: SteamClient) -> None:
    responses.get(APP_DETAILS_URL, status=429)
    responses.get(APP_DETAILS_URL, body="null")
    responses.get(APP_DETAILS_URL, json={"10": {"success": True, "data": {"name": "X"}}})
    assert client.get_app_details(10) == {"name": "X"}
    assert len(responses.calls) == 3


def _sleep_recorder() -> tuple[list[float], SteamClient]:
    sleeps: list[float] = []
    client = SteamClient(
        request_interval=0, max_retries=3, throttle_cooldown=60.0, sleep=sleeps.append
    )
    return sleeps, client


@responses.activate
def test_throttling_waits_at_least_the_cooldown() -> None:
    sleeps, client = _sleep_recorder()
    responses.get(APP_DETAILS_URL, status=429)
    responses.get(APP_DETAILS_URL, status=503)
    responses.get(APP_DETAILS_URL, json={"10": {"success": True, "data": {"name": "X"}}})
    assert client.get_app_details(10) == {"name": "X"}
    # 429 -> cooldown instead of 2s; 503 keeps the plain exponential backoff.
    assert sleeps == [60.0, 4.0]


@responses.activate
def test_throttling_honors_longer_retry_after() -> None:
    sleeps, client = _sleep_recorder()
    responses.get(APP_DETAILS_URL, status=429, headers={"Retry-After": "90"})
    responses.get(APP_DETAILS_URL, status=429, headers={"Retry-After": "soon"})
    responses.get(APP_DETAILS_URL, json={"10": {"success": False}})
    assert client.get_app_details(10) is None
    assert sleeps == [90.0, 60.0]


@responses.activate
def test_unsuccessful_app_details_returns_none(client: SteamClient) -> None:
    responses.get(APP_DETAILS_URL, json={"10": {"success": False}})
    assert client.get_app_details(10) is None


@responses.activate
def test_gives_up_after_max_retries_without_leaking_key(client: SteamClient) -> None:
    responses.get(APP_LIST_URL, body=requests.ConnectionError("boom ?key=test-key"))
    with pytest.raises(SteamApiError) as exc:
        client.get_app_list()
    assert "test-key" not in str(exc.value)
    assert len(responses.calls) == 3  # 1 + max_retries


@responses.activate
def test_non_retryable_status_fails_fast(client: SteamClient) -> None:
    responses.get(APP_DETAILS_URL, status=400)
    with pytest.raises(SteamApiError):
        client.get_app_details(10)
    assert len(responses.calls) == 1


def test_request_pacing() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    class Session:
        def get(self, *args, **kwargs):
            now[0] += 0.5
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b'{"10": {"success": false}}'
            return resp

    client = SteamClient(request_interval=2.0, session=Session(), sleep=sleep, clock=lambda: now[0])
    client.get_app_details(10)
    client.get_app_details(10)
    assert sleeps == [1.5]


@responses.activate
def test_iter_pages_stops_at_cutoff(client: SteamClient) -> None:
    responses.get(
        REVIEWS_URL,
        json={
            "success": 1,
            "cursor": "c1",
            "query_summary": {"total_reviews": 3},
            "reviews": [_review(3, 300), _review(2, 200), _review(1, 100)],
        },
    )
    pages = list(client.iter_review_pages(10, since_ts=150))
    assert [[r["recommendationid"] for r in p.reviews] for p in pages] == [["3", "2"]]
    assert pages[0].total_reviews == 3
    assert len(responses.calls) == 1


@responses.activate
def test_iter_pages_full_history_until_repeated_cursor(client: SteamClient) -> None:
    responses.get(
        REVIEWS_URL,
        match=[matchers.query_param_matcher({"cursor": "*"}, strict_match=False)],
        json={
            "success": 1,
            "cursor": "c1",
            "query_summary": {"total_reviews": 2},
            "reviews": [_review(2, 200)],
        },
    )
    responses.get(
        REVIEWS_URL,
        match=[matchers.query_param_matcher({"cursor": "c1"}, strict_match=False)],
        json={"success": 1, "cursor": "c1", "reviews": [_review(1, 100)]},
    )
    pages = list(client.iter_review_pages(10))
    assert [len(p.reviews) for p in pages] == [1, 1]
    assert pages[1].total_reviews is None
    assert len(responses.calls) == 2
    assert "filter=recent" in responses.calls[0].request.url


@responses.activate
def test_iter_pages_respects_max_reviews(client: SteamClient) -> None:
    responses.get(
        REVIEWS_URL,
        json={
            "success": 1,
            "cursor": "c1",
            "query_summary": {"total_reviews": 9},
            "reviews": [_review(3, 300), _review(2, 200), _review(1, 100)],
        },
    )
    pages = list(client.iter_review_pages(10, max_reviews=2))
    assert sum(len(p.reviews) for p in pages) == 2
    assert len(responses.calls) == 1


@responses.activate
def test_iter_pages_no_new_reviews_still_reports_total(client: SteamClient) -> None:
    responses.get(
        REVIEWS_URL,
        json={
            "success": 1,
            "cursor": "c1",
            "query_summary": {"total_reviews": 1},
            "reviews": [_review(1, 100)],
        },
    )
    pages = list(client.iter_review_pages(10, since_ts=100))
    assert len(pages) == 1 and pages[0].reviews == [] and pages[0].total_reviews == 1
