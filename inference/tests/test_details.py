from decimal import Decimal

import boto3
from conftest import DETAILS_TABLE, FIRST_GAME, N_GAMES, FakeSource, details_table

from steam_inference.contracts import GameDetails, clean_text
from steam_inference.details import read_game_details, sync_game_details
from steam_inference.writer import DynamoWriter


def _writer():
    return DynamoWriter(DETAILS_TABLE, region="us-east-1", concurrency=2, key="game_id")


def _items() -> dict[int, dict]:
    table = boto3.resource("dynamodb").Table(DETAILS_TABLE)
    return {int(item["game_id"]): item for item in table.scan()["Items"]}


def test_clean_text():
    assert clean_text(" Shoot &amp; <b>loot</b>\n now. ") == "Shoot & loot now."
    assert clean_text("  <br> ") is None
    assert clean_text(None) is None


def test_read_game_details(source):
    details = {d.game_id: d for d in read_game_details(source, 7)}
    assert len(details) == N_GAMES
    odd = details[1000 + FIRST_GAME + 1]
    assert odd.short_description == f"Shoot & loot {FIRST_GAME + 1}."
    assert odd.categories == [] and odd.genres == ["Action", "Indie"]
    assert details[1000 + FIRST_GAME].short_description is None


def test_item_has_no_nulls_and_decimal_price():
    item = GameDetails(game_id=10, name="A", price=9.989).to_item()
    assert item == {
        "game_id": 10,
        "name": "A",
        "price": Decimal("9.99"),
        "developers": [],
        "publishers": [],
        "genres": [],
        "categories": [],
    }


def test_sync_is_insert_only(aws, marts):
    assert sync_game_details(FakeSource(marts), _writer()) == (N_GAMES, 7)
    items = _items()
    assert len(items) == N_GAMES
    assert items[1000 + FIRST_GAME]["header_image"] == f"https://cdn/{1000 + FIRST_GAME}.jpg"
    assert "short_description" not in items[1000 + FIRST_GAME]

    # details are static: stored games are never rewritten, new games are added
    boto3.resource("dynamodb").Table(DETAILS_TABLE).update_item(
        Key={"game_id": 1000 + FIRST_GAME},
        UpdateExpression="SET #n = :n",
        ExpressionAttributeNames={"#n": "name"},
        ExpressionAttributeValues={":n": "kept"},
    )
    grown = {**marts, "game_details": details_table(range(FIRST_GAME, FIRST_GAME + N_GAMES + 1))}
    assert sync_game_details(FakeSource(grown), _writer()) == (1, 7)
    items = _items()
    assert len(items) == N_GAMES + 1 and items[1000 + FIRST_GAME]["name"] == "kept"


def test_missing_mart_skips(aws, marts):
    without = {k: v for k, v in marts.items() if k != "game_details"}
    assert sync_game_details(FakeSource(without), _writer()) == (0, None)
    assert _items() == {}
