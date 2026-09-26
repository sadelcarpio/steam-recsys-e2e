import gzip
import io
import json
from datetime import UTC, datetime

import boto3
import numpy as np
from conftest import BUCKET, TABLE

from steam_inference.contracts import ONLINE_BUNDLE_ARRAYS, OnlineBundleManifest, SearchIndex
from steam_inference.features import load_inference_data
from steam_inference.online import S3BundleStore
from steam_inference.pipeline import run_inference
from steam_inference.search import build_search_index, encode_search_index, index_from_catalog
from steam_inference.writer import DynamoWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)


def test_pipeline_publishes_a_pinned_catalog(settings, source, store, champion):
    boto3.client("s3").put_bucket_versioning(
        Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"}
    )
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.online_bundle == f"s3://{BUCKET}/serving/online/manifest.json"

    s3 = boto3.client("s3")
    manifest = OnlineBundleManifest.model_validate(
        json.loads(s3.get_object(Bucket=BUCKET, Key="serving/online/manifest.json")["Body"].read())
    )
    assert (manifest.model_id, manifest.catalog_games) == ("abc123", 20)
    # the model's own numpy user tower (training wrote it with the model)
    assert manifest.user_tower_key == "models/abc123/user_tower.npz"
    s3.head_object(Bucket=BUCKET, Key=manifest.user_tower_key)
    assert manifest.catalog_version_id  # pinned: the bucket is versioned
    obj = s3.get_object(
        Bucket=BUCKET, Key=manifest.catalog_key, VersionId=manifest.catalog_version_id
    )
    with np.load(io.BytesIO(obj["Body"].read()), allow_pickle=False) as npz:
        assert tuple(npz.files) == ONLINE_BUNDLE_ARRAYS
        assert npz["item_embeddings"].shape == (20, champion.config.output_dim)
        assert np.allclose(np.linalg.norm(npz["item_embeddings"], axis=1), 1.0, atol=1e-5)

    # a second run rewrites both; the manifest follows the new version
    run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    newer = OnlineBundleManifest.model_validate_json(
        s3.get_object(Bucket=BUCKET, Key="serving/online/manifest.json")["Body"].read()
    )
    assert newer.catalog_version_id != manifest.catalog_version_id


def test_model_without_numpy_user_tower_publishes_nothing(settings, source, store, champion):
    boto3.client("s3").delete_object(Bucket=BUCKET, Key="models/abc123/user_tower.npz")
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert not summary.skipped and summary.written == 5  # the batch run still completes
    assert summary.online_bundle is None
    listed = boto3.client("s3").list_objects_v2(Bucket=BUCKET, Prefix="serving/")
    assert listed["KeyCount"] == 0


def test_skipped_run_publishes_nothing(settings, source, store):
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles)
    assert summary.skipped and summary.online_bundle is None
    listed = boto3.client("s3").list_objects_v2(Bucket=BUCKET, Prefix="serving/")
    assert listed["KeyCount"] == 0


def test_pipeline_publishes_the_search_index(settings, source, store, champion):
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.search_index == f"s3://{BUCKET}/serving/search/games.json"

    obj = boto3.client("s3").get_object(Bucket=BUCKET, Key="serving/search/games.json")
    assert (obj["ContentType"], obj["ContentEncoding"]) == ("application/json", "gzip")
    index = SearchIndex.model_validate_json(gzip.decompress(obj["Body"].read()))
    assert (index.model_id, index.generated_at) == ("abc123", NOW)
    # the same games as the online catalog, most reviewed first
    assert len(index.games) == summary.catalog_games == 20
    reviews = [n for _, _, n in index.games]
    assert reviews == sorted(reviews, reverse=True) and reviews[0] > 0


def test_search_index_ranks_by_loaded_reviews_then_appid(settings, source):
    data = load_inference_data(source)
    index = build_search_index(data, model_id="m", now=NOW)
    counts = np.bincount(data.reviews.game_idx, minlength=len(data.games.catalog.row_of))
    expected = sorted(
        (
            (-int(counts[idx]), int(gid), str(name))
            for gid, idx, name in zip(
                data.games.game_id,
                data.games.catalog.items.game_idx,
                data.games.name,
                strict=True,
            )
        ),
    )
    assert index.games == [(gid, name, -neg) for neg, gid, name in expected]
    assert gzip.decompress(encode_search_index(index)) == index.model_dump_json().encode()


def test_model_without_numpy_user_tower_publishes_no_search_index(
    settings, source, store, champion
):
    boto3.client("s3").delete_object(Bucket=BUCKET, Key="models/abc123/user_tower.npz")
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.search_index is None


def test_index_from_a_published_catalog_matches_the_run(settings, source, store, champion):
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    s3 = boto3.client("s3")
    index_obj = s3.get_object(Bucket=BUCKET, Key="serving/search/games.json")
    published = SearchIndex.model_validate_json(gzip.decompress(index_obj["Body"].read()))
    manifest = OnlineBundleManifest.model_validate_json(
        s3.get_object(Bucket=BUCKET, Key="serving/online/manifest.json")["Body"].read()
    )
    catalog = s3.get_object(Bucket=BUCKET, Key=manifest.catalog_key)["Body"].read()
    with np.load(io.BytesIO(catalog)) as npz:
        arrays = {name: npz[name] for name in npz.files}
    # counts per appid as the run saw them (scripts/publish_search_index.py gets them from Athena)
    counts = {game_id: n for game_id, _, n in published.games}
    rebuilt = index_from_catalog(arrays, counts, model_id="abc123", now=NOW)
    assert rebuilt == published
    # without counts: every game, 0 reviews, by appid
    bare = index_from_catalog(arrays, {}, model_id="abc123", now=NOW)
    assert [g for g, _, _ in bare.games] == sorted(g for g, _, _ in published.games)
    assert {n for _, _, n in bare.games} == {0}
