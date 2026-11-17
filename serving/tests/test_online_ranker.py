"""Spec 9, phase 5: the serving pair's ranker re-orders POST /recommendations (numpy only).
Parity with training's features and LightGBM: inference's tests/test_online_ranker.py."""

import boto3
import numpy as np
import pytest
from conftest import (
    BUNDLE_BUCKET,
    BUNDLE_PREFIX,
    DETAILS_TABLE,
    FREE_GAMES,
    RECS_TABLE,
    body,
    make_ranker,
    publish_bundle,
)
from test_app_online import post

from steam_serving.app import App
from steam_serving.online import BundleLoader
from steam_serving.ranker import OnlineRanker, TreeEnsemble
from steam_serving.repository import DynamoRepository


@pytest.fixture
def s3(aws):
    client = boto3.client("s3")
    client.create_bucket(Bucket=BUNDLE_BUCKET)
    client.put_bucket_versioning(
        Bucket=BUNDLE_BUCKET, VersioningConfiguration={"Status": "Enabled"}
    )
    return client


def _app(settings) -> App:
    loader = BundleLoader(BUNDLE_BUCKET, BUNDLE_PREFIX, region="us-east-1", refresh_seconds=300)
    return App(
        settings, DynamoRepository(RECS_TABLE, DETAILS_TABLE, region="us-east-1"), loader.get
    )


def _recommend(settings, limit=8) -> dict:
    return body(post(_app(settings), {"liked_game_ids": [12, 11], "limit": limit}))


def test_the_ranker_orders_the_online_list(tables, settings, s3):
    publish_bundle(s3, ranker=make_ranker())
    data = _recommend(settings)
    assert data["ranker_id"] == "k1" and data["source"] == "online"
    games = [r["game_id"] for r in data["recommendations"]]
    free = [g in FREE_GAMES for g in games]
    assert free == sorted(free, reverse=True) and any(free)  # free games first
    assert [r["rank"] for r in data["recommendations"]] == list(range(1, 9))
    assert not {12, 11} & set(games)


def test_without_a_ranker_or_in_shadow_the_order_is_retrieval(tables, settings, s3):
    publish_bundle(s3)
    plain = _recommend(settings)
    assert "ranker_id" not in plain  # null fields are left out
    scores = [r["score"] for r in plain["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    publish_bundle(s3, ranker=make_ranker(), ranker_mode="shadow")
    shadow = _recommend(settings)
    assert "ranker_id" not in shadow and shadow["recommendations"] == plain["recommendations"]


@pytest.mark.parametrize(
    "ranker",
    [
        make_ranker(model_id="other"),  # exported for another retriever
        make_ranker(ranker_id="k2"),  # not the ranker the manifest names
        make_ranker(schema=1),  # an older feature schema
        b"not an npz",
    ],
)
def test_an_unusable_ranker_keeps_retrieval_order(tables, settings, s3, ranker):
    publish_bundle(s3)
    plain = _recommend(settings)
    publish_bundle(s3, ranker=ranker)
    data = _recommend(settings)
    assert "ranker_id" not in data and data["recommendations"] == plain["recommendations"]


def test_a_ranker_error_on_a_request_keeps_retrieval_order(tables, settings, s3, monkeypatch):
    publish_bundle(s3)
    plain = _recommend(settings)
    publish_bundle(s3, ranker=make_ranker())

    def broken(self, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(OnlineRanker, "features", broken)
    data = _recommend(settings)
    assert "ranker_id" not in data and data["recommendations"] == plain["recommendations"]


def test_candidates_beyond_the_limit_are_ranked():
    """The ranker re-orders its `candidates` retrieved games, then the list is cut: a free game
    retrieved below the limit can move up."""
    ranker = OnlineRanker.from_bytes(make_ranker(candidates=30))
    candidates = np.array([1, 2, 4, 5, 3])  # row r is game 10 + r: rows 2 and 5 are free
    order = ranker.order(np.array([0]), candidates, np.linspace(1, 0.5, 5), 1_790_000_000)
    assert candidates[order][0] == 2  # free games first, each group in retrieval order
    assert candidates[order].tolist() == [2, 5, 1, 4, 3]


def test_tree_evaluator_missing_values_follow_the_default_child():
    ranker = OnlineRanker.from_bytes(make_ranker())
    trees: TreeEnsemble = ranker.trees
    x = np.zeros((3, 15))
    x[:, 7] = [0.0, 1.0, np.nan]  # is_free: not free, free, missing (default left)
    assert trees.predict(x).tolist() == [0.0, 1.0, 0.0]
