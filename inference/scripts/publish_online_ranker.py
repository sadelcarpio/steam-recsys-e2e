"""Publish the serving pair's ranker for POST /recommendations (serving/online/ranker.npz) next to
the online catalog that is already in S3, without an inference run (spec 9, phase 5).

An inference run does this with every catalog it publishes; this script covers the time until
the next one (a ranker promoted after the last run, or a mode change). It reads no
`interactions`: only the game marts (`game_features` latest rows, tags, `lkp_games`,
`game_details`) and the `game_popularity_weekly` mart, popularity as of now. The ranker is the
serving pair's (`evaluation/champion/pair.json`), and only when its retriever is the model of
the published catalog. The catalog is not touched: the manifest is rewritten with the same
catalog version plus the ranker's fields.

    uv run python scripts/publish_online_ranker.py --dry-run          # ./ranker.npz + a check
    uv run python scripts/publish_online_ranker.py --mode on          # uploads

Before uploading it checks the file with serving's own code (`steam_serving`, a dev dependency)
and prints one example list in retrieval and ranker order. Run it after any inference run in
progress has finished (that run rewrites the manifest), and once the inference image that
publishes the ranker itself is deployed, so the next run keeps it.

Needs s3:GetObject / PutObject on serving/online/*, GetObject on models/<id>/user_tower.npz,
rankers/champion/* and evaluation/champion/pair.json, and Glue / S3 reads of the marts.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
from steam_training.artifacts import ArtifactStore
from steam_training.data import IcebergSource

from steam_inference.contracts import OnlineBundleManifest
from steam_inference.features import (
    TAG_TABLES,
    _optional_snapshots,
    load_games,
)
from steam_inference.online import MANIFEST_NAME, RankerPayload, S3BundleStore
from steam_inference.ranking import load_serving_ranker, online_ranker

log = logging.getLogger("publish_online_ranker")

GAME_TABLES = ["lkp_games", "game_features", "lkp_genres", "lkp_developers"]


def check(
    s3,
    bucket: str,
    manifest: OnlineBundleManifest,
    catalog: bytes,
    ranker: RankerPayload,
    liked: list[int],
) -> None:
    """Load the result with serving's code (as the Lambda will) and print one list both ways."""
    try:
        from steam_serving.online import BundleManifest, OnlineModel, UserTower
        from steam_serving.ranker import OnlineRanker
    except ImportError:
        log.warning("steam_serving not installed (dev dependency): check skipped")
        return
    fields = {"ranker_id": ranker.ranker_id, "ranker_mode": "on", "ranker_key": "ranker.npz"}
    served = BundleManifest.model_validate(manifest.model_dump() | fields)
    tower = UserTower.from_bytes(
        s3.get_object(Bucket=bucket, Key=manifest.user_tower_key)["Body"].read()
    )
    numpy_ranker = OnlineRanker.from_bytes(ranker.payload)  # raises on a format / schema problem
    ranked = OnlineModel.from_bytes(served, catalog, tower, numpy_ranker)  # checks rows / ids
    plain = OnlineModel.from_bytes(served, catalog, tower)
    with np.load(io.BytesIO(catalog), allow_pickle=False) as npz:
        game_ids = npz["item_game_id"]
    if not liked:  # the most reviewed game of the last 30 days
        recent = numpy_ranker.popularity[:, 1]
        liked = [int(game_ids[int(np.argmax(recent))])]
    a, b = plain.recommend(liked, 10), ranked.recommend(liked, 10)
    if not b.game_ids:
        log.warning("none of %s is in the catalog: no example", liked)
        return
    log.info("example for liked games %s (used %s):", liked, b.used_game_ids)
    for i, (x, y) in enumerate(zip(a.names, b.names, strict=False), start=1):
        log.info("  %2d. %-40.40s | %s", i, x, y)
    log.info("retrieval order (left) vs ranker %s (right): checks passed", b.ranker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", help="default: model-artifacts-<account id>")
    parser.add_argument("--online-prefix", default="serving/online")
    parser.add_argument("--database", default="steam_marts")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--mode", choices=["on", "shadow"], default="on", help="on: re-order; shadow: log only"
    )
    parser.add_argument("--candidates", type=int, default=100, help="RETRIEVAL_CANDIDATES")
    parser.add_argument("--liked", type=int, nargs="*", default=[], help="appids for the example")
    parser.add_argument("--cache-dir", help="DATA_CACHE_DIR: keep the mart files read on disk")
    parser.add_argument("--dry-run", action="store_true", help="write ./ranker.npz instead")
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
    catalog = s3.get_object(Bucket=bucket, Key=manifest.catalog_key, **extra)["Body"].read()
    with np.load(io.BytesIO(catalog), allow_pickle=False) as npz:
        game_idx = npz["item_game_idx"].astype(np.int64)
    log.info(
        "published catalog: model %s, %d games, generated %s (ranker in manifest: %s)",
        manifest.model_id,
        len(game_idx),
        manifest.generated_at,
        manifest.ranker_id,
    )

    source = IcebergSource(args.database, args.region, cache_dir=args.cache_dir)
    snapshots = {table: source.snapshot_id(table) for table in GAME_TABLES}
    snapshots |= _optional_snapshots(source, TAG_TABLES)
    if not set(TAG_TABLES) <= set(snapshots):
        log.warning("tag marts missing: tag_cosine will be 0 for every game")
    games = load_games(source, snapshots)
    if game_idx.max(initial=0) >= len(games.catalog.row_of):
        sys.exit("the published catalog has games beyond lkp_games: run inference instead")

    now = datetime.now(UTC).replace(tzinfo=None)  # naive UTC, as inference's runs
    store = ArtifactStore(bucket, s3)
    ranker, reason = load_serving_ranker(source, store, games.catalog, manifest.model_id, as_of=now)
    if ranker is None:
        sys.exit(f"no ranker to publish: {reason}")
    log.info("ranker: %s", reason)
    payload = online_ranker(
        ranker,
        game_idx,
        retriever_id=manifest.model_id,
        candidates=args.candidates,
        mode=args.mode,
        as_of=now,
    )
    check(s3, bucket, manifest, catalog, payload, args.liked)

    if args.dry_run:
        Path("ranker.npz").write_bytes(payload.payload)
        log.info(
            "dry run: wrote ranker.npz (%.1f MB), nothing uploaded", len(payload.payload) / 1e6
        )
        return
    uri = S3BundleStore(bucket, prefix, region=args.region).publish_ranker(manifest, payload)
    log.info("published: %s (the Lambda picks it up within ONLINE_REFRESH_SECONDS)", uri)


if __name__ == "__main__":
    main()
