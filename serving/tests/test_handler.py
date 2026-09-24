from conftest import body, event

from steam_serving import handler


def test_handler_builds_the_app_once(tables, monkeypatch):
    handler.app.cache_clear()
    monkeypatch.setenv("DEFAULT_LIMIT", "3")
    first = handler.handler(event("/popular"), None)
    assert first["statusCode"] == 200 and len(body(first)["recommendations"]) == 3
    assert handler.app() is handler.app()
    handler.app.cache_clear()


def test_handler_serves_online_from_s3(tables, monkeypatch):
    import json

    import boto3
    from conftest import BUNDLE_BUCKET, publish_bundle

    s3 = boto3.client("s3")
    s3.create_bucket(Bucket=BUNDLE_BUCKET)
    s3.put_bucket_versioning(Bucket=BUNDLE_BUCKET, VersioningConfiguration={"Status": "Enabled"})
    publish_bundle(s3)
    handler.app.cache_clear()
    monkeypatch.setenv("MODEL_ARTIFACTS_BUCKET", BUNDLE_BUCKET)
    request = event("/recommendations", method="POST")
    request["body"] = json.dumps({"liked_game_ids": [12], "limit": 2})
    response = handler.handler(request, None)
    assert response["statusCode"] == 200 and body(response)["source"] == "online"
    handler.app.cache_clear()


def test_handler_without_bucket_disables_online(tables):
    handler.app.cache_clear()
    request = event("/recommendations", method="POST")
    request["body"] = '{"liked_game_ids": [12]}'
    assert handler.handler(request, None)["statusCode"] == 503
    handler.app.cache_clear()
