import boto3
import pytest
from conftest import BUNDLE_BUCKET, publish_user_index

from steam_serving.users import UserIndex

IDS = [4559049361332948866, 2**63 - 1, 0, 42]


@pytest.fixture
def s3(aws):
    client = boto3.client("s3")
    client.create_bucket(Bucket=BUNDLE_BUCKET)
    client.put_bucket_versioning(
        Bucket=BUNDLE_BUCKET, VersioningConfiguration={"Status": "Enabled"}
    )
    return client


def _index(refresh_seconds: int = 300) -> UserIndex:
    return UserIndex(
        BUNDLE_BUCKET, "serving/users", region="us-east-1", refresh_seconds=refresh_seconds
    )


def test_each_user_is_one_ranged_read(s3):
    publish_user_index(s3, IDS)
    users = _index()
    manifest = users.manifest()
    assert manifest.max_user == 4 and manifest.index_version_id
    assert [users.user_id(manifest, i) for i in range(1, 5)] == [str(i) for i in IDS]
    for outside in (0, 5):
        with pytest.raises(ValueError):
            users.user_id(manifest, outside)


def test_reads_stay_on_the_manifest_version(s3):
    publish_user_index(s3, IDS)
    users = _index(refresh_seconds=3600)
    manifest = users.manifest()
    # a newer index.bin without its manifest yet: the pinned version is still read
    s3.put_object(Bucket=BUNDLE_BUCKET, Key="serving/users/index.bin", Body=bytes(32))
    assert users.manifest() is manifest
    assert users.user_id(manifest, 1) == str(IDS[0])


def test_refresh_picks_up_a_new_run(s3):
    publish_user_index(s3, IDS)
    users = _index(refresh_seconds=0)
    assert users.manifest().max_user == 4
    publish_user_index(s3, [7, 8])
    manifest = users.manifest()
    assert manifest.max_user == 2 and users.user_id(manifest, 2) == "8"


def test_nothing_published_and_failed_refreshes(s3):
    users = _index(refresh_seconds=0)
    assert users.manifest() is None
    publish_user_index(s3, IDS)
    good = users.manifest()
    s3.put_object(Bucket=BUNDLE_BUCKET, Key="serving/users/index.json", Body=b"{not json")
    assert users.manifest() == good  # a broken refresh keeps the cached manifest
    fresh = _index()
    with pytest.raises(ValueError):
        fresh.manifest()  # nothing cached: the error surfaces (a 500)


def test_unknown_format_is_refused(s3):
    publish_user_index(s3, IDS)
    body = s3.get_object(Bucket=BUNDLE_BUCKET, Key="serving/users/index.json")["Body"].read()
    s3.put_object(
        Bucket=BUNDLE_BUCKET,
        Key="serving/users/index.json",
        Body=body.replace(b'"format_version": 1', b'"format_version": 2'),
    )
    with pytest.raises(ValueError, match="format 2"):
        _index().manifest()
