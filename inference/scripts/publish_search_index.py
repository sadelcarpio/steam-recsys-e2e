"""Publish the frontend's search index (serving/search/games.json) from the online catalog that
is already in S3, without an inference run. Review counts per game come from one Athena query
over the `interactions` mart (as a run would count them with MAX_USERS=0).

Adult games (steam_inference.adult: name or "Sexual Content" / "Nudity" genre from the
`game_details` mart) are left out of the index and, when the catalog still has them, the
catalog is republished without them too (so POST /recommendations can't return them). Batch
recommendations already in DynamoDB are only cleaned by the next inference run.

    uv run python scripts/publish_search_index.py --dry-run   # writes ./games.json.gz only
    uv run python scripts/publish_search_index.py             # uploads to s3://model-artifacts-<acct>/

Needs s3:GetObject on serving/online/*, Athena on the workgroup and s3:PutObject on
serving/online/* and the index key. The next inference run rewrites both with the same rules.
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

from steam_inference.adult import ADULT_GENRES, ADULT_NAME
from steam_inference.contracts import OnlineBundleManifest
from steam_inference.online import MANIFEST_NAME, S3BundleStore, encode_catalog, filter_catalog
from steam_inference.search import encode_search_index, index_from_catalog

log = logging.getLogger("publish_search_index")

REVIEWS_QUERY = "SELECT game_idx, count(*) AS reviews FROM {database}.interactions GROUP BY 1"
ADULT_GENRE_QUERY = (
    "SELECT DISTINCT game_id FROM {database}.game_details "
    "WHERE cardinality(filter(game_genres, g -> g IN ({genres}))) > 0"
)


def athena_rows(athena, s3, query: str, *, work_group: str) -> list[dict[str, str]]:
    execution = athena.start_query_execution(QueryString=query, WorkGroup=work_group)[
        "QueryExecutionId"
    ]
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution)["QueryExecution"]
        state = status["Status"]["State"]
        if state not in ("QUEUED", "RUNNING"):
            break
        time.sleep(2)
    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena query {state}: {status['Status'].get('StateChangeReason')}")
    # The result CSV is faster to read than paginated GetQueryResults.
    location = status["ResultConfiguration"]["OutputLocation"]
    bucket, key = location.removeprefix("s3://").split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return list(csv.DictReader(io.StringIO(body)))


def catalog_names(arrays: dict[str, np.ndarray]) -> list[str]:
    utf8, offsets = arrays["item_name_utf8"].tobytes(), arrays["item_name_offsets"]
    return [utf8[offsets[i] : offsets[i + 1]].decode() for i in range(len(offsets) - 1)]


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
    athena = boto3.client("athena", region_name=args.region)
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

    genres = ", ".join(f"'{g}'" for g in sorted(ADULT_GENRES))
    adult_genre_ids = {
        int(r["game_id"])
        for r in athena_rows(
            athena,
            s3,
            ADULT_GENRE_QUERY.format(database=args.database, genres=genres),
            work_group=args.work_group,
        )
    }
    names = catalog_names(arrays)
    adult = np.array(
        [
            int(g) in adult_genre_ids or bool(ADULT_NAME.search(n))
            for g, n in zip(arrays["item_game_id"], names, strict=True)
        ]
    )
    log.info("adult games in the catalog: %d", int(adult.sum()))
    catalog_changed = bool(adult.any())
    if catalog_changed:
        arrays = filter_catalog(arrays, ~adult)

    reviews: dict[int, int] = {}
    if not args.no_reviews:
        rows = athena_rows(
            athena,
            s3,
            REVIEWS_QUERY.format(database=args.database),
            work_group=args.work_group,
        )
        per_idx = {int(r["game_idx"]): int(r["reviews"]) for r in rows if r["game_idx"]}
        reviews = {
            int(game_id): per_idx.get(int(idx), 0)
            for game_id, idx in zip(arrays["item_game_id"], arrays["item_game_idx"], strict=True)
        }

    index = index_from_catalog(arrays, reviews, model_id=manifest.model_id, now=datetime.now(UTC))
    body = encode_search_index(index)
    log.info("search index: %d games, %.1f MB gzipped", len(index.games), len(body) / 1e6)
    if args.dry_run:
        with open("games.json.gz", "wb") as f:
            f.write(body)
        log.info("dry run: wrote games.json.gz, catalog not republished")
        return
    store = S3BundleStore(bucket, prefix, region=args.region, search_key=args.search_key)
    if catalog_changed:
        uri = store.publish(
            encode_catalog(arrays),
            model_id=manifest.model_id,
            generated_at=manifest.generated_at,
            catalog_games=len(arrays["item_game_id"]),
            user_tower_key=manifest.user_tower_key,
        )
        log.info("republished the catalog without adult games: %s", uri)
    log.info("published %s", store.publish_search_index(body))


if __name__ == "__main__":
    main()
