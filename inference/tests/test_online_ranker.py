"""Spec 9, phase 5: the ranker in the serving Lambda. Serving's numpy port (`steam_serving.ranker`)
must give LightGBM's scores on the exported trees and `compute_features`' features on the
exported game side; the inference run publishes it with the catalog, in the mode of the run."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from conftest import FakeSource
from steam_serving import ranker as serving_ranker
from steam_serving.online import BundleManifest, OnlineModel, UserTower
from steam_serving.ranker import OnlineRanker, TreeEnsemble
from steam_training.ranker.export import (
    encode_ranker,
    game_arrays,
    ranker_arrays,
    tree_arrays,
)
from steam_training.ranker.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    POPULARITY_FEATURES,
    GameTable,
    PopularityTable,
    compute_features,
    week_of,
)
from steam_training.ranker.model import FitResult
from test_ranking import NOW, ranker_marts, serve_ranker

from steam_inference.online import LocalBundleStore
from steam_inference.pipeline import run_inference
from steam_inference.writer import JsonlWriter


def test_serving_port_constants_match_training():
    assert serving_ranker.FEATURES == FEATURES
    assert serving_ranker.POPULARITY_FEATURES == POPULARITY_FEATURES
    assert serving_ranker.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION


def _features(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, len(FEATURES)))
    x[rng.random(x.shape) < 0.15] = np.nan  # missing values: NaN splits
    x[:, FEATURES.index("is_free")] = rng.integers(0, 2, n)  # zeros
    return x


@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_tree_evaluator_matches_lightgbm(zero_as_missing):
    rng = np.random.default_rng(3)
    x = _features(rng, 600)
    label = (np.nan_to_num(x[:, 3]) + np.nan_to_num(x[:, 10]) > 0.5).astype(int)
    subset = [f for f in FEATURES if f != "publisher_share"]  # an ablation-style subset
    idx = [FEATURES.index(f) for f in subset]
    booster = lgb.train(
        {
            "objective": "lambdarank",
            "verbose": -1,
            "min_data_in_leaf": 5,
            "num_leaves": 15,
            "zero_as_missing": zero_as_missing,
        },
        lgb.Dataset(x[:, idx], label, group=[20] * 30, feature_name=subset),
        num_boost_round=40,
    )
    fit = FitResult(booster, subset, best_iteration=30)  # fewer trees than trained
    trees = TreeEnsemble.from_arrays(tree_arrays(fit))
    assert trees.num_trees == 30
    probe = _features(rng, 300)
    probe[:5] = 0.0  # exact zeros everywhere
    expected = booster.predict(probe[:, idx], num_iteration=30)
    np.testing.assert_allclose(trees.predict(probe), expected, rtol=0, atol=1e-9)


def _game_side(rng: np.random.Generator, num_games: int):
    tags = rng.random((num_games, 6)).astype(np.float32) * (rng.random((num_games, 6)) < 0.5)
    tags[3] = 0  # a game without tags
    tags /= np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1e-12)
    games = GameTable(
        tags=tags,
        developer=rng.integers(-1, 3, num_games),
        publisher=rng.integers(-1, 3, num_games),
        genres=(rng.random((num_games, 4)) < 0.4).astype(np.float32),
        categories=(rng.random((num_games, 3)) < 0.5).astype(np.float32),
        is_free=rng.integers(0, 2, num_games).astype(np.float32),
        log_price=np.where(rng.random(num_games) < 0.2, np.nan, rng.random(num_games) * 4).astype(
            np.float32
        ),
        release_day=np.where(
            rng.random(num_games) < 0.2, np.nan, 19_000 + rng.integers(0, 900, num_games)
        ).astype(np.float32),
    )
    weeks = 20
    first_week = 2_800
    game = np.repeat(np.arange(num_games), weeks)
    week = np.tile(np.arange(first_week, first_week + weeks), num_games)
    reviews = rng.integers(0, 9, len(game))
    positives = rng.binomial(reviews, 0.7)
    popularity = PopularityTable.from_weekly(
        game, week, positives, reviews, num_games, first_week, first_week + weeks - 1
    )
    return games, popularity


@pytest.mark.parametrize("history_size", [0, 1, 4])
def test_online_features_match_compute_features(history_size):
    rng = np.random.default_rng(history_size)
    num_games = 30
    games, popularity = _game_side(rng, num_games)
    # the published catalog: a permutation of games 2.. (0 / 1 are padding / OOV)
    catalog = rng.permutation(np.arange(2, num_games))
    as_of = int((2_812 * 7 - 3) * 86_400 + 3 * 86_400 + 5_000)  # inside week 2812
    assert week_of(as_of) == 2_812
    arrays = game_arrays(games, popularity, catalog, as_of)
    online = OnlineRanker.from_bytes(encode_ranker(_with_trees(arrays, history_length=3)))

    history_rows = rng.choice(len(catalog), history_size, replace=False)
    candidate_rows = rng.choice(np.setdiff1d(np.arange(len(catalog)), history_rows), 8, False)
    scores = np.sort(rng.random(8).astype(np.float32))[::-1]
    now = as_of + 3 * 86_400  # same week: popularity unchanged, age moves on
    got = online.features(history_rows, candidate_rows, scores, now)

    history = np.zeros((1, 20), dtype=np.int64)  # padding
    used = catalog[history_rows][:3]  # the ranker reads its history_length
    history[0, : len(used)] = used
    expected = compute_features(
        history,
        catalog[candidate_rows][None],
        scores[None],
        np.array([now]),
        games,
        popularity,
    )[0]
    # popularity from the export week; the same week as `now`, so batch agrees
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


def _with_trees(game: dict, *, history_length: int) -> dict:
    """game_arrays + metadata + a one-leaf tree: what ranker_arrays writes."""
    meta = {
        "format_version": np.asarray(1),
        "feature_schema_version": np.asarray(FEATURE_SCHEMA_VERSION),
        "features_utf8": np.frombuffer(",".join(FEATURES).encode(), np.uint8),
        "popularity_features_utf8": np.frombuffer(",".join(POPULARITY_FEATURES).encode(), np.uint8),
        "ranker_id": np.asarray("k1"),
        "model_id": np.asarray("abc123"),
        "history_length": np.asarray(history_length),
        "candidates": np.asarray(10),
        "as_of_seconds": np.asarray(0),
    }
    trees = {
        "tree_roots": np.zeros(1, np.int32),
        "node_feature": np.full(1, -1, np.int32),
        "node_threshold": np.zeros(1),
        "node_left": np.zeros(1, np.int32),
        "node_right": np.zeros(1, np.int32),
        "node_default_left": np.zeros(1, bool),
        "node_missing": np.zeros(1, np.int8),
        "node_value": np.zeros(1),
        "max_depth": np.asarray(0),
    }
    return meta | trees | game


def test_a_ranker_of_another_feature_schema_is_refused():
    rng = np.random.default_rng(0)
    games, popularity = _game_side(rng, 6)
    arrays = _with_trees(game_arrays(games, popularity, np.arange(2, 6), 0), history_length=3)
    arrays["feature_schema_version"] = np.asarray(FEATURE_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="feature schema"):
        OnlineRanker.from_bytes(encode_ranker(arrays))


def _publish(settings, store, tmp_path, mode: str) -> Path:
    directory = tmp_path / mode
    run_inference(
        settings.model_copy(
            update={"ranker_mode": mode, "retrieval_candidates": 10, "ranker_top_k": 5}
        ),
        FakeSource(ranker_marts()),
        store,
        JsonlWriter(str(tmp_path / f"{mode}.jsonl")),
        None,
        bundle_store=LocalBundleStore(str(directory)),
        now=NOW,
        save_online_lists=False,
    )
    return directory


def _online(store, directory: Path) -> OnlineModel:
    manifest = BundleManifest.model_validate_json((directory / "manifest.json").read_text())
    tower = UserTower.from_bytes(store._get(manifest.user_tower_key))
    ranker = None
    if manifest.ranker_key:
        ranker = OnlineRanker.from_bytes((directory / manifest.ranker_key).read_bytes())
    return OnlineModel.from_bytes(manifest, (directory / "catalog.npz").read_bytes(), tower, ranker)


def test_inference_publishes_the_ranker_and_serving_applies_it(settings, store, champion, tmp_path):
    serve_ranker(store)  # LightGBM model scoring free games first
    directory = _publish(settings, store, tmp_path, "on")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["ranker_id"] == "k1" and manifest["ranker_mode"] == "on"
    assert manifest["ranker_key"] == "ranker.npz" and manifest["ranker_sha256"]
    online = _online(store, directory)
    assert online.ranker is not None and online.ranker.candidates == 10

    liked = [1002, 1004]  # game_idx 2, 4
    result = online.recommend(liked, 5)
    assert result.ranker_id == "k1" and len(result.game_ids) == 5
    free = [(g - 1000) % 3 == 0 for g in result.game_ids]  # game_idx % 3 == 0 is free
    assert free == sorted(free, reverse=True)
    assert not set(liked) & set(result.game_ids)


def test_shadow_publishes_the_ranker_but_serves_retrieval_order(
    settings, store, champion, tmp_path
):
    serve_ranker(store)
    online = _online(store, _publish(settings, store, tmp_path, "shadow"))
    assert online.ranker_mode == "shadow"
    retrieval = OnlineModel(online.manifest, _catalog(online), online.user_tower)
    shadow = online.recommend([1002, 1004], 5)
    assert shadow.ranker_id is None
    assert shadow.game_ids == retrieval.recommend([1002, 1004], 5).game_ids


def _catalog(online: OnlineModel) -> dict[str, np.ndarray]:
    return {
        "item_embeddings": online.item_embeddings,
        "item_game_id": online.item_game_id,
        "item_game_idx": online._row_game_idx,
        "item_name_utf8": np.frombuffer(online._name_bytes, np.uint8),
        "item_name_offsets": online._name_offsets,
    }


def test_without_a_serving_ranker_the_manifest_names_none(settings, store, champion, tmp_path):
    online = _online(store, _publish(settings, store, tmp_path, "on"))  # no pair.json ranker
    assert online.ranker is None and online.manifest.ranker_id is None
    assert online.recommend([1002], 5).ranker_id is None


def test_ranker_arrays_name_the_pair(settings, store, champion):
    rng = np.random.default_rng(1)
    games, popularity = _game_side(rng, 8)
    x = rng.random((40, len(FEATURES)))
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "min_data_in_leaf": 3},
        lgb.Dataset(x, x[:, 0], feature_name=FEATURES),
        num_boost_round=3,
    )
    arrays = ranker_arrays(
        FitResult(booster, FEATURES, 3),
        ranker_id="k9",
        model_id="abc123",
        history_length=20,
        candidates=100,
        games=games,
        popularity=popularity,
        game_idx=np.arange(2, 8),
        as_of_seconds=1_700_000_000,
    )
    online = OnlineRanker.from_bytes(encode_ranker(arrays))
    assert (online.ranker_id, online.model_id, online.candidates) == ("k9", "abc123", 100)
    assert online.rows == 6 and online.trees.num_trees == 3


def _script():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "publish_online_ranker.py"
    spec = importlib.util.spec_from_file_location("publish_online_ranker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_script_adds_the_ranker_to_the_published_catalog(
    settings, store, champion, tmp_path, monkeypatch, caplog
):
    """A catalog published without a ranker (an older run), then a pair promotion: the script
    adds ranker.npz and the manifest's ranker fields, the catalog version stays, and serving's
    loader applies it."""
    from steam_serving.online import BundleLoader

    from steam_inference.online import S3BundleStore

    bundles = S3BundleStore(store.bucket, "serving/online", region="us-east-1")
    run_inference(
        settings.model_copy(update={"ranker_mode": "off"}),
        FakeSource(ranker_marts()),
        store,
        JsonlWriter(str(tmp_path / "off.jsonl")),
        None,
        bundle_store=bundles,
        now=NOW,
        save_online_lists=False,
    )
    key = "serving/online/manifest.json"
    before = json.loads(store.s3.get_object(Bucket=store.bucket, Key=key)["Body"].read())
    assert before["ranker_id"] is None
    serve_ranker(store)

    script = _script()
    monkeypatch.setattr(script, "IcebergSource", lambda *a, **k: FakeSource(ranker_marts()))
    monkeypatch.setattr(
        "sys.argv",
        ["publish_online_ranker.py", "--bucket", store.bucket, "--liked", "1002", "1004"],
    )
    with caplog.at_level("INFO"):
        script.main()
    assert "checks passed" in caplog.text

    after = json.loads(store.s3.get_object(Bucket=store.bucket, Key=key)["Body"].read())
    assert after["ranker_id"] == "k1" and after["ranker_mode"] == "on"
    assert after["catalog_version_id"] == before["catalog_version_id"]
    assert after["catalog_sha256"] == before["catalog_sha256"]

    model = BundleLoader(
        store.bucket, "serving/online", region="us-east-1", refresh_seconds=0
    ).get()
    assert model.ranker is not None and model.ranker_mode == "on"
    result = model.recommend([1002, 1004], 5)
    assert result.ranker_id == "k1"
    free = [(g - 1000) % 3 == 0 for g in result.game_ids]
    assert free == sorted(free, reverse=True)


def test_publish_script_refuses_without_a_serving_ranker(
    settings, store, champion, tmp_path, monkeypatch
):
    from steam_inference.online import S3BundleStore

    run_inference(
        settings.model_copy(update={"ranker_mode": "off"}),
        FakeSource(ranker_marts()),
        store,
        JsonlWriter(str(tmp_path / "off.jsonl")),
        None,
        bundle_store=S3BundleStore(store.bucket, "serving/online", region="us-east-1"),
        now=NOW,
        save_online_lists=False,
    )
    script = _script()
    monkeypatch.setattr(script, "IcebergSource", lambda *a, **k: FakeSource(ranker_marts()))
    monkeypatch.setattr("sys.argv", ["publish_online_ranker.py", "--bucket", store.bucket])
    with pytest.raises(SystemExit, match="no ranker to publish"):
        script.main()
