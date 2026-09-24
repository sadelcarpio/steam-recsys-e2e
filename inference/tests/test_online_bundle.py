import io
import json
from datetime import UTC, datetime

import boto3
import numpy as np
from conftest import BUCKET, TABLE

from steam_inference.contracts import ONLINE_BUNDLE_ARRAYS, OnlineBundleManifest
from steam_inference.online import S3BundleStore
from steam_inference.pipeline import run_inference
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
