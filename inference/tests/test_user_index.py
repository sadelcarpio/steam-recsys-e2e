import json
from datetime import UTC, datetime

import boto3
import numpy as np
import pyarrow as pa
import pytest
from conftest import BUCKET, REVIEWS, TABLE, FakeSource

from steam_inference.contracts import UserIndexManifest
from steam_inference.features import load_inference_data, review_activity
from steam_inference.online import LocalBundleStore, S3BundleStore
from steam_inference.pipeline import run_inference, select_rerank_users
from steam_inference.retrieval import retrieve
from steam_inference.user_index import encode_user_index, load_user_index
from steam_inference.writer import DynamoWriter

NOW = datetime(2024, 7, 1, tzinfo=UTC)
# REVIEWS: 101 (7 reviews), 104 (18), 102 (6), 103 (2), 105 (1)
RANKED = [104, 101, 102, 103, 105]


def test_rows_are_placed_by_user_idx(source):
    assert load_user_index(source).tolist() == RANKED


def test_missing_or_empty_mart_publishes_nothing(marts):
    assert (
        load_user_index(FakeSource({k: v for k, v in marts.items() if k != "user_index"})) is None
    )
    empty = marts["user_index"].slice(0, 0)
    assert load_user_index(FakeSource({**marts, "user_index": empty})) is None


@pytest.mark.parametrize("user_idx", [[1, 2, 4], [0, 1, 2], [1, 1, 2]])
def test_a_gap_or_duplicate_fails(marts, user_idx):
    table = pa.table({"user_idx": user_idx, "user_id": [7, 8, 9]})
    with pytest.raises(ValueError, match="not exactly"):
        load_user_index(FakeSource({**marts, "user_index": table}))


def test_encoding_maps_user_idx_to_a_byte_offset():
    payload = encode_user_index(np.array(RANKED, dtype=np.int64))
    assert len(payload) == 8 * len(RANKED)
    i = 3  # user_idx 3
    assert int.from_bytes(payload[(i - 1) * 8 : i * 8], "little", signed=True) == RANKED[i - 1]


def test_inference_order_is_the_index_order(settings, source, champion, store):
    """The top of the index are the users MAX_USERS keeps and the reranked ones."""
    activity = review_activity(source, 7)
    for n in range(1, len(REVIEWS) + 1):
        positive_ranked = [u for u in RANKED if any(p for _, p in REVIEWS[u])]
        assert sorted(activity.most_active(n).tolist()) == sorted(positive_ranked[:n])
    data = load_inference_data(source)
    model, _ = store.load_model("champion")
    candidates = retrieve(
        model, data.games, data.users, data.reviews, k=5, user_batch_size=2, item_batch_size=7
    )
    everyone = settings.model_copy(update={"rerank_min_reviews": 1})
    rerank_order = data.users.user_id[select_rerank_users(data, candidates, everyone)].tolist()
    assert rerank_order == [u for u in RANKED if u in rerank_order]


def test_pipeline_publishes_a_pinned_index(settings, source, store, champion):
    s3 = boto3.client("s3")
    s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    bundles = S3BundleStore(BUCKET, "serving/online", region="us-east-1")
    writer = DynamoWriter(TABLE, region="us-east-1", concurrency=2)
    summary = run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert summary.user_index == f"s3://{BUCKET}/serving/users/index.json"

    def manifest() -> UserIndexManifest:
        body = s3.get_object(Bucket=BUCKET, Key="serving/users/index.json")["Body"].read()
        return UserIndexManifest.model_validate(json.loads(body))

    first = manifest()
    assert (first.max_user, first.index_key) == (5, "serving/users/index.bin")
    assert first.index_version_id  # pinned: the bucket is versioned
    # serving's read: one user by byte range, at the pinned version
    obj = s3.get_object(
        Bucket=BUCKET, Key=first.index_key, VersionId=first.index_version_id, Range="bytes=8-15"
    )
    assert int.from_bytes(obj["Body"].read(), "little", signed=True) == RANKED[1]

    run_inference(settings, source, store, writer, None, bundle_store=bundles, now=NOW)
    assert manifest().index_version_id != first.index_version_id


def test_local_dry_run_writes_the_index(tmp_path):
    store = LocalBundleStore(str(tmp_path))
    path = store.publish_user_index(b"\x01" + bytes(7), max_user=1, generated_at=NOW)
    manifest = UserIndexManifest.model_validate_json((tmp_path / "index.json").read_text())
    assert path.endswith("index.json") and manifest.index_version_id is None
    assert (tmp_path / "index.bin").read_bytes() == b"\x01" + bytes(7)
