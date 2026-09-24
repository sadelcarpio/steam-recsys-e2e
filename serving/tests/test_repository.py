from conftest import DETAILS_TABLE, RECS_TABLE

from steam_serving.repository import DynamoRepository


def _repo():
    return DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1")


def test_recommendations(tables):
    stored = _repo().recommendations("76561198000000002")
    assert [r.game_id for r in stored.recommendations] == [10, 11, 12]
    assert _repo().recommendations("1") is None


def test_games_in_one_batch_skip_missing_and_duplicates(tables):
    games = _repo().games([10, 11, 10, 105, 39])
    assert set(games) == {10, 11, 39}
    assert games[11].short_description == "About 11."


def test_games_retries_unprocessed_keys(tables, monkeypatch):
    repo = _repo()
    real = repo.resource.batch_get_item
    calls = []

    def throttled(RequestItems):  # noqa: N803
        calls.append(len(RequestItems[DETAILS_TABLE]["Keys"]))
        if len(calls) == 1:  # first call: everything comes back unprocessed
            return {"Responses": {}, "UnprocessedKeys": RequestItems}
        return real(RequestItems=RequestItems)

    monkeypatch.setattr(repo.resource, "batch_get_item", throttled)
    monkeypatch.setattr("steam_serving.repository.time.sleep", lambda s: None)
    assert set(repo.games([10, 11])) == {10, 11}
    assert calls == [2, 2]
