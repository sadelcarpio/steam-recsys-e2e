from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from steam_ingestion.seed_coming_soon import (
    ComingSoon,
    looks_released,
    parse_release_date,
    query_coming_soon,
    seed,
)
from steam_ingestion.state import GameIdsState

TODAY = date(2026, 9, 29)


def test_parse_release_date_only_accepts_day_precise_dates() -> None:
    assert parse_release_date("Sep 25, 2026") == date(2026, 9, 25)
    assert parse_release_date("25 Sep, 2026") == date(2026, 9, 25)
    assert parse_release_date("October 1, 2026") == date(2026, 10, 1)
    for vague in ("Coming soon", "To be announced", "Q4 2026", "2027", "October 2026", "", None):
        assert parse_release_date(vague) is None


def test_looks_released() -> None:
    assert looks_released(ComingSoon(1, "Sep 25, 2026", False), TODAY)
    assert not looks_released(ComingSoon(1, "Sep 29, 2026", False), TODAY)  # today: not yet
    assert not looks_released(ComingSoon(1, "Q3 2026", False), TODAY)
    assert looks_released(ComingSoon(1, "Coming soon", True), TODAY)  # reviews exist


def test_seed_sets_the_flag_and_requeues_released_games(aws: SimpleNamespace) -> None:
    for appid in (1, 2, 3):
        aws.games.put_item(Item={"appid": appid, "status": "scraped", "attempts": 0})
    # 3 was already scraped by a spec-10 scraper: never overwritten
    aws.games.update_item(
        Key={"appid": 3},
        UpdateExpression="SET coming_soon = :c",
        ExpressionAttributeValues={":c": False},
    )
    state = GameIdsState(aws.games)
    games = [
        ComingSoon(1, "Coming soon", False),
        ComingSoon(2, "Sep 1, 2026", False),
        ComingSoon(3, "2027", False),
        ComingSoon(99, "2027", False),  # no state item: nothing created
    ]

    def requeue(appid: int) -> bool:
        return state.requeue([appid]) == 1

    counts = seed(games, lambda a: state.seed_coming_soon(a, True), requeue, TODAY)

    assert counts == {"seeded": 2, "skipped": 2, "requeued": 1}
    loaded = state.load_all()
    assert loaded[1].coming_soon and loaded[1].status == "scraped"
    assert loaded[2].coming_soon and loaded[2].status == "pending"
    assert not loaded[3].coming_soon
    assert 99 not in loaded
    # idempotent: nothing left to seed or re-queue
    again = seed(games, lambda a: state.seed_coming_soon(a, True), requeue, TODAY)
    assert again == {"skipped": 4}


class FakeAthena:
    def __init__(self) -> None:
        self.query = ""

    def start_query_execution(self, QueryString: str, WorkGroup: str) -> dict:
        self.query = QueryString
        return {"QueryExecutionId": "q1"}

    def get_query_execution(self, QueryExecutionId: str) -> dict:
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_paginator(self, name: str) -> SimpleNamespace:
        def row(*values: str | None) -> dict:
            return {"Data": [{"VarCharValue": v} if v is not None else {} for v in values]}

        rows = [row("appid", "release_date", "has_reviews"), row("7", None, "true")]
        return SimpleNamespace(paginate=lambda **_: [{"ResultSet": {"Rows": rows}}])


def test_query_coming_soon_parses_nulls() -> None:
    athena = FakeAthena()
    assert query_coming_soon(athena, "wg", "steam_raw") == [ComingSoon(7, None, True)]
    assert "from steam_raw.games" in athena.query
