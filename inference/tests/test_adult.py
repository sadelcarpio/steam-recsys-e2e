import gzip
from datetime import UTC, datetime

import boto3
import numpy as np
import pyarrow as pa
import pytest
from conftest import BUCKET, FIRST_GAME, TABLE, FakeSource

from steam_inference.adult import adult_mask, is_adult
from steam_inference.contracts import POPULAR_USER_ID, SearchIndex
from steam_inference.features import load_inference_data
from steam_inference.online import S3BundleStore, filter_catalog
from steam_inference.pipeline import rerank_request, run_inference
from steam_inference.retrieval import retrieve
from steam_inference.writer import DynamoWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)
# game_idx 3: explicit name; game_idx 4 (genre id FIRST_GAME + 4 % 4 = 2 -> "Nudity"),
# 8, 12, ...: every game of that genre
NAMED, GENRE_ID = 3, FIRST_GAME


@pytest.mark.parametrize(
    ("name", "genres", "adult"),
    [
        ("Hentai Girl Linda", [], True),
        ("My Sexy Waitress", [], True),
        ("NSFW Puzzle 18+", [], True),
        ("Pornstar Simulator", [], True),
        ("A calm game", ["Indie", "Nudity"], True),
        ("A calm game", ["Sexual Content"], True),
        ("Portal 2", ["Action"], False),
        ("HENTAISLAND: Lost Pantsu", [], True),
        ("FutaPunk 2069", [], True),
        ("Boobie Jump", [], True),
        ("Comic Strip Hero", [], False),
        ("Fantasy Tavern Sextet", [], False),
        ("Sussex Tales", [], False),  # a word containing "sex"
        ("CONTROL Resonant", ["Action"], False),
        ("Essex Farm Simulator", [], False),
    ],
)
def test_is_adult(name, genres, adult):
    assert is_adult(name, genres) is adult


@pytest.fixture
def adult_marts(marts):
    features = marts["game_features"].to_pylist()
    for row in features:
        if row["game_idx"] == NAMED:
            row["game_name"] = "Hentai Game"
    marts["game_features"] = pa.Table.from_pylist(features)
    genres = marts["lkp_genres"].to_pylist()
    for row in genres:
        if row["id"] == GENRE_ID:
            row["name"] = "Nudity"
    marts["lkp_genres"] = pa.Table.from_pylist(genres)
    return marts


def _adult_game_ids(data) -> set[int]:
    return {int(g) for g in data.games.game_id[adult_mask(data.games)]}


def test_mask_by_name_and_genre(adult_marts):
    data = load_inference_data(FakeSource(adult_marts))
    idx = data.games.catalog.items.game_idx
    expected = (idx == NAMED) | (idx % 4 == 0)  # genre FIRST_GAME + idx % 4 == GENRE_ID
    assert np.array_equal(adult_mask(data.games), expected)


def test_excluded_games_are_never_retrieved(adult_marts, champion):
    data = load_inference_data(FakeSource(adult_marts))
    excluded = adult_mask(data.games)
    candidates = retrieve(
        champion,
        data.games,
        data.users,
        data.reviews,
        k=30,
        user_batch_size=2,
        item_batch_size=4,
        excluded=excluded,
    )
    valid = candidates.rows[candidates.rows >= 0]
    assert len(valid) and not excluded[valid].any()


def test_pipeline_publishes_nothing_adult(settings, adult_marts, store, champion):
    source = FakeSource(adult_marts)
    data = load_inference_data(source)
    adult = _adult_game_ids(data)
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.adult_games == len(adult) > 0

    items = boto3.resource("dynamodb").Table(TABLE).scan()["Items"]
    recommended = {int(r["game_id"]) for i in items for r in i["recommendations"]}
    assert recommended and not recommended & adult
    assert any(i["user_id"] == POPULAR_USER_ID for i in items)

    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=BUCKET, Key="serving/search/games.json")["Body"].read()
    searchable = {g for g, _, _ in SearchIndex.model_validate_json(gzip.decompress(body)).games}
    assert len(searchable) == summary.catalog_games - len(adult)
    assert not searchable & adult


def test_prompt_leaves_out_liked_adult_games(adult_marts, champion):
    data = load_inference_data(FakeSource(adult_marts))
    excluded = adult_mask(data.games)
    candidates = retrieve(
        champion, data.games, data.users, data.reviews, k=5, user_batch_size=8, item_batch_size=8
    )
    user = int(np.flatnonzero(data.users.user_id == 101)[0])  # liked game_idx 3 ("Hentai Game")
    assert any("Hentai" in line for line in rerank_request(data, candidates, user).liked)
    request = rerank_request(data, candidates, user, excluded)
    assert request.liked and not any("Hentai" in line for line in request.liked)


def test_filter_catalog_keeps_names_aligned():
    names = ["A", "Hentai B", "Ç"]
    utf8 = b"".join(n.encode() for n in names)
    offsets = np.array([0, 1, 1 + len(b"Hentai B"), len(utf8)], dtype=np.int64)
    arrays = {
        "item_embeddings": np.eye(3, dtype=np.float32),
        "item_game_id": np.array([1, 2, 3]),
        "item_game_idx": np.array([10, 20, 30]),
        "item_name_utf8": np.frombuffer(utf8, dtype=np.uint8),
        "item_name_offsets": offsets,
    }
    out = filter_catalog(arrays, np.array([True, False, True]))
    assert out["item_game_id"].tolist() == [1, 3]
    assert out["item_embeddings"].tolist() == [[1, 0, 0], [0, 0, 1]]
    got = out["item_name_utf8"].tobytes()
    o = out["item_name_offsets"]
    assert [got[o[i] : o[i + 1]].decode() for i in range(2)] == ["A", "Ç"]
