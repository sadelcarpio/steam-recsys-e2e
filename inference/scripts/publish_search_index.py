"""Publish the frontend's search index (serving/search/games.json) from the online catalog that
is already in S3, without an inference run. Review counts per game come from one Athena query
over the `interactions` mart (as a run would count them with MAX_USERS=0).

    uv run python scripts/publish_search_index.py --dry-run   # writes ./games.json.gz only
    uv run python scripts/publish_search_index.py             # uploads to s3://model-artifacts-<acct>/

Needs s3:GetObject on serving/online/*, Athena on the workgroup and s3:PutObject on the index key.
The next inference run overwrites the index with the same content (and fresher counts).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import time
from datetime import UTC, datetime

import boto3
import numpy as np

from steam_inference.contracts import OnlineBundleManifest
from steam_inference.online import MANIFEST_NAME, S3BundleStore
from steam_inference.search import encode_search_index, index_from_catalog

log = logging.getLogger("publish_search_index")

REVIEWS_QUERY = "SELECT game_idx, count(*) AS reviews FROM {database}.interactions GROUP BY 1"


def reviews_per_game_idx(athena, s3, *, database: str, work_group: str) -> dict[int, int]:
    execution = athena.start_query_execution(
        QueryString=REVIEWS_QUERY.format(database=database), WorkGroup=work_group
    )["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution)["QueryExecution"]
        state = status["Status"]["State"]
        if state not in ("QUEUED", "RUNNING"):
            break
        time.sleep(2)
    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena query {state}: {status['Status'].get('StateChangeReason')}")
    # The result CSV (one row per game) is faster to read than paginated GetQueryResults.
    location = status["ResultConfiguration"]["OutputLocation"]
    bucket, key = location.removeprefix("s3://").split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    rows = csv.DictReader(io.StringIO(body))
    return {int(r["game_idx"]): int(r["reviews"]) for r in rows if r["game_idx"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", help="default: model-artifacts-<account id>")
    parser.add_argument("--online-prefix", default="serving/online")
    parser.add_argument("--search-key", default="serving/search/games.json")
    parser.add_argument("--database", default="steam_marts")
    parser.add_argument("--work-group", default="steam-recsys-etl")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--no-reviews", action="store_true", help="skip Athena: every count 0")
    parser.add_argument("--dry-run", action="store_true", help="write ./games.json.gz instead")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s3 = boto3.client("s3", region_name=args.region)
    if args.bucket:
        bucket = args.bucket
    else:
        account = boto3.client("sts", region_name=args.region).get_caller_identity()["Account"]
        bucket = f"model-artifacts-{account}"
    prefix = args.online_prefix.strip("/")
    manifest = OnlineBundleManifest.model_validate(
        json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST_NAME}")["Body"].read())
    )
    extra = {"VersionId": manifest.catalog_version_id} if manifest.catalog_version_id else {}
    payload = s3.get_object(Bucket=bucket, Key=manifest.catalog_key, **extra)["Body"].read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    log.info("catalog of model %s: %d games", manifest.model_id, manifest.catalog_games)

    reviews: dict[int, int] = {}
    if not args.no_reviews:
        per_idx = reviews_per_game_idx(
            boto3.client("athena", region_name=args.region),
            s3,
            database=args.database,
            work_group=args.work_group,
        )
        reviews = {
            int(game_id): per_idx.get(int(idx), 0)
            for game_id, idx in zip(arrays["item_game_id"], arrays["item_game_idx"], strict=True)
        }
        log.info("review counts for %d games", sum(1 for n in reviews.values() if n))

    index = index_from_catalog(arrays, reviews, model_id=manifest.model_id, now=datetime.now(UTC))
    body = encode_search_index(index)
    log.info("search index: %d games, %.1f MB gzipped", len(index.games), len(body) / 1e6)
    if args.dry_run:
        with open("games.json.gz", "wb") as f:
            f.write(body)
        log.info("dry run: wrote games.json.gz; top 5: %s", index.games[:5])
    else:
        store = S3BundleStore(bucket, prefix, region=args.region, search_key=args.search_key)
        log.info("published %s", store.publish_search_index(body))


if __name__ == "__main__":
    main()
