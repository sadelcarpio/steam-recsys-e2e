import boto3
import numpy as np
import pytest
from conftest import (
    BUNDLE_BUCKET,
    BUNDLE_PREFIX,
    make_catalog,
    make_manifest,
    make_user_tower,
    publish_bundle,
)

from steam_serving.online import BundleLoader, BundleManifest, OnlineModel, UserTower


def _reference_user(tower: UserTower, history: list[int]) -> np.ndarray:
    """The user tower written out step by step (model.py, UserTower)."""
    ids = [h if h < len(tower.game_table) else 1 for h in history]
    ids = [i if (tower.seen_games[i] or i == 0) else 1 for i in ids]
    present = [i for i in ids if i != 0]
    pooled = np.mean(tower.game_table[present], axis=0) if present else np.zeros(6)
    x = np.concatenate([pooled, [len(present) / 5]])
    out = np.maximum(x @ tower.w1.T + tower.b1, 0) @ tower.w2.T + tower.b2
    return out / np.linalg.norm(out)


@pytest.mark.parametrize(
    "history",
    [[2, 3, 4, 0, 0], [7, 0, 0, 0, 0], [5, 30, 31, 2, 0], [0, 0, 0, 0, 0], [2, 3, 4, 6, 8]],
)
def test_user_embedding_matches_the_reference(online_model, history):
    tower = online_model.user_tower
    got = tower.embed(np.array(history))
    np.testing.assert_allclose(got, _reference_user(tower, history), rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-5)


def test_rows_of(online_model):
    assert online_model.rows_of(np.array([10, 39, 9, 40, 25])).tolist() == [0, 29, -1, -1, 15]


def test_recommend_excludes_liked_and_ranks_by_score(online_model):
    liked = [12, 11, 12, 99, 10, 14, 15, 16, 17]  # 12 twice, 99 unknown
    result = online_model.recommend(liked, k=5)
    assert result.ignored_game_ids == [99]
    assert result.used_game_ids == [12, 11, 10, 14, 15]  # first 5 known, most recent first
    assert not set(result.game_ids) & set(liked)  # all liked games excluded, not only the used
    assert result.names == [f"Game {g} é" for g in result.game_ids]
    history = np.array([4, 3, 2, 6, 7])  # game_idx of the used games
    scores = online_model.item_embeddings @ online_model.user_tower.embed(history)
    allowed = [r for r in range(30) if 10 + r not in liked]
    best = sorted(allowed, key=lambda r: -scores[r])[:5]
    assert result.game_ids == [10 + r for r in best]
    assert result.scores == [round(float(scores[r]), 4) for r in best]


def test_recommend_caps_k_at_the_remaining_catalog(online_model):
    liked = list(range(10, 37))  # 27 of 30 games
    assert len(online_model.recommend(liked, k=10).game_ids) == 3


def test_recommend_unknown_games_only(online_model):
    result = online_model.recommend([1, 2], k=5)
    assert (result.game_ids, result.ignored_game_ids) == ([], [1, 2])


def test_from_bytes_checks_hash_format_and_model():
    catalog, tower = make_catalog(), UserTower.from_bytes(make_user_tower())
    manifest = make_manifest(catalog)
    with pytest.raises(ValueError, match="sha256"):
        OnlineModel.from_bytes(BundleManifest.model_validate(manifest), catalog + b"x", tower)
    newer = BundleManifest.model_validate({**manifest, "format_version": 3})
    with pytest.raises(ValueError, match="format"):
        OnlineModel.from_bytes(newer, catalog, tower)
    other = BundleManifest.model_validate(make_manifest(catalog, model_id="def456"))
    with pytest.raises(ValueError, match="user tower of abc123"):
        OnlineModel.from_bytes(other, catalog, tower)


def test_user_tower_checks_its_format():
    import io

    with np.load(io.BytesIO(make_user_tower())) as npz:
        arrays = {name: npz[name] for name in npz.files}
    buffer = io.BytesIO()
    np.savez(buffer, **{**arrays, "format_version": np.int64(2)})
    with pytest.raises(ValueError, match="user tower format"):
        UserTower.from_bytes(buffer.getvalue())


@pytest.fixture
def s3(aws):
    client = boto3.client("s3")
    client.create_bucket(Bucket=BUNDLE_BUCKET)
    client.put_bucket_versioning(
        Bucket=BUNDLE_BUCKET, VersioningConfiguration={"Status": "Enabled"}
    )
    return client


def _loader(refresh_seconds=300):
    return BundleLoader(
        BUNDLE_BUCKET, BUNDLE_PREFIX, region="us-east-1", refresh_seconds=refresh_seconds
    )


def test_loader_without_a_bundle(s3):
    assert _loader().get() is None


def test_loader_loads_the_pinned_version(s3):
    manifest = publish_bundle(s3, seed=1)
    # a newer catalog object without its manifest yet: the loader must keep the pinned version
    s3.put_object(Bucket=BUNDLE_BUCKET, Key=manifest["catalog_key"], Body=make_catalog(seed=2))
    model = _loader().get()
    assert model.manifest.catalog_sha256 == manifest["catalog_sha256"]


def test_loader_reuses_the_user_tower_of_the_same_model(s3, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("steam_serving.online.time.monotonic", lambda: clock[0])
    publish_bundle(s3, seed=1)
    loader = _loader(refresh_seconds=0)
    first = loader.get()
    publish_bundle(s3, seed=2)  # next weekly run: new catalog, same model
    clock[0] += 1
    second = loader.get()
    assert second is not first and second.user_tower is first.user_tower


def test_loader_caches_then_refreshes(s3, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("steam_serving.online.time.monotonic", lambda: clock[0])
    publish_bundle(s3, seed=1, model_id="m1")
    loader = _loader(refresh_seconds=300)
    first = loader.get()
    assert first.manifest.model_id == "m1"
    publish_bundle(s3, seed=2, model_id="m2")
    clock[0] += 10
    assert loader.get() is first  # within the refresh interval
    clock[0] += 300
    assert loader.get().manifest.model_id == "m2"
    clock[0] += 300
    second = loader.get()
    clock[0] += 300
    assert loader.get() is second  # same manifest: not reloaded


def test_failed_refresh_keeps_the_model(s3, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("steam_serving.online.time.monotonic", lambda: clock[0])
    publish_bundle(s3, seed=1, model_id="m1")
    loader = _loader(refresh_seconds=0)
    assert loader.get().manifest.model_id == "m1"
    s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{BUNDLE_PREFIX}/manifest.json", Body=b"{broken")
    clock[0] += 1
    assert loader.get().manifest.model_id == "m1"


def test_broken_first_load_raises(s3):
    s3.put_object(Bucket=BUNDLE_BUCKET, Key=f"{BUNDLE_PREFIX}/manifest.json", Body=b"{broken")
    with pytest.raises(ValueError):
        _loader().get()
