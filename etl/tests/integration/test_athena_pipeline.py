"""End-to-end run of the dbt project on real Athena + Iceberg.

Scenario, in isolated Glue databases `<ci_id>_{raw,staging,intermediate,marts}` and under
`s3://$ETL_CI_BUCKET/<ci_id>/`:
  1. batch 1 raw parquet -> dbt run              (snapshot `batch1`)
  2. dbt run again, no new data                  (snapshot `rerun`, must equal `batch1`)
  3. batch 2 raw parquet -> dbt run              (snapshot `batch2`, incremental)
  4. dbt build --full-refresh                    (snapshot `full`, must equal `batch2`)
Athena time is per query, not per row, so for speed the dbt data tests only run in step 4
(steps 1-3 are checked by the assertions below and by `batch2 == full`), Iceberg maintenance is
off, dbt uses 8 threads and snapshots query tables in parallel.
Everything is dropped afterwards (set ETL_CI_KEEP=true to keep it for debugging).

Needs AWS credentials (the CI OIDC role) and ETL_CI_BUCKET, ETL_CI_WORK_GROUP, AWS_REGION.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import boto3
import pyarrow.parquet as pq
import pytest
from pyathena import connect
from pyathena.cursor import DictCursor

from steam_etl.config import EtlSettings
from steam_etl.contracts import MART_CONTRACTS
from steam_etl.runner import run_dbt
from tests.integration import fixtures as fx

pytestmark = pytest.mark.athena

REQUIRED_ENV = ("ETL_CI_BUCKET", "ETL_CI_WORK_GROUP", "AWS_REGION")
if missing := [k for k in REQUIRED_ENV if not os.environ.get(k)]:
    pytest.skip(f"Athena integration test needs {', '.join(missing)}", allow_module_level=True)

LAYERS = ("raw", "staging", "intermediate", "marts")
TABLES = {
    "intermediate": [
        "int_games__deduplicated",
        "int_reviews__deduplicated",
        "int_review_events",
        "int_game_review_counts",
    ],
    "marts": list(MART_CONTRACTS),
}
KEYS = {
    "int_games__deduplicated": ("game_name_key",),
    "int_reviews__deduplicated": ("review_id",),
    "int_review_events": ("review_id",),
    "int_game_review_counts": ("game_id", "timestamp"),
    "lkp_developers": ("id",),
    "lkp_publishers": ("id",),
    "lkp_genres": ("id",),
    "lkp_categories": ("id",),
    "lkp_games": ("game_idx",),
    "game_features": ("game_id", "timestamp"),
    "user_features": ("user_id", "timestamp"),
    "interactions": ("review_id",),
}
Snapshot = dict[str, list[dict[str, Any]]]


def ts(offset: int) -> str:
    """Athena's varchar rendering of timestamp(6) T0 + offset seconds."""
    return datetime.fromtimestamp(fx.T0 + offset, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


EPOCH = "1970-01-01 00:00:00.000000"


class Lakehouse:
    def __init__(self) -> None:
        self.region = os.environ["AWS_REGION"]
        self.bucket = os.environ["ETL_CI_BUCKET"]
        self.work_group = os.environ["ETL_CI_WORK_GROUP"]
        run = os.environ.get("GITHUB_RUN_ID")
        suffix = (
            f"{run}_{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}" if run else uuid.uuid4().hex[:8]
        )
        self.ci_id = f"ci_{suffix}"
        self.prefix = f"{self.ci_id}/"
        self.s3 = boto3.client("s3", region_name=self.region)
        self.glue = boto3.client("glue", region_name=self.region)
        self.conn = connect(
            work_group=self.work_group,
            region_name=self.region,
            s3_staging_dir=f"s3://{self.bucket}/{self.prefix}athena-results/",
            cursor_class=DictCursor,
        )
        self.settings = EtlSettings(
            athena_work_group=self.work_group,
            athena_s3_staging_dir=f"s3://{self.bucket}/{self.prefix}athena-results/",
            iceberg_s3_data_dir=f"s3://{self.bucket}/{self.prefix}iceberg/",
            dbt_schema=self.ci_id,
            aws_region=self.region,
            dbt_threads=8,
            iceberg_maintenance=False,
        )

    def db(self, layer: str) -> str:
        return f"{self.ci_id}_{layer}"

    # ---- setup / teardown -------------------------------------------------------------------

    def create_raw_tables(self) -> None:
        # Prod registers these in Terraform (infrastructure/etl.tf); same columns.
        self.query(f"create database if not exists `{self.db('raw')}`")
        for table in fx.RAW_SCHEMAS:
            location = f"s3://{self.bucket}/{self.prefix}raw/{table}/"
            self.query(fx.external_table_ddl(self.db("raw"), table, location))

    def upload(self, name: str, games: list[dict], reviews: list[dict]) -> None:
        for table, rows in (("games", games), ("reviews", reviews)):
            buf = io.BytesIO()
            pq.write_table(fx.to_table(rows, fx.RAW_SCHEMAS[table]), buf, compression="zstd")
            key = f"{self.prefix}raw/{table}/{name}.parquet"
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def teardown(self) -> None:
        for layer in LAYERS:
            with contextlib.suppress(self.glue.exceptions.EntityNotFoundException):
                self.glue.delete_database(Name=self.db(layer))
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})

    # ---- dbt / queries ------------------------------------------------------------------------

    def dbt(self, command: str, full_refresh: bool = False) -> None:
        settings = self.settings.model_copy(update={"full_refresh": full_refresh})
        result = run_dbt(settings, command)
        assert result.success, "\n".join(result.failures)

    def query(self, sql: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall()) if cur.description else []

    def fetch(self, layer: str, table: str, columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """All rows; arrays as lists (via JSON), timestamps as varchar."""
        select, arrays = [], []
        for col in columns:
            name, dtype = col["column_name"], col["data_type"]
            if dtype.startswith("array"):
                select.append(f'json_format(cast("{name}" as json)) as "{name}"')
                arrays.append(name)
            elif dtype.startswith("timestamp"):
                select.append(f'cast("{name}" as varchar) as "{name}"')
            else:
                select.append(f'"{name}"')
        rows = self.query(f'select {", ".join(select)} from "{self.db(layer)}"."{table}"')
        for row in rows:
            for name in arrays:
                row[name] = json.loads(row[name]) if row[name] is not None else None
        return sorted(rows, key=lambda r: tuple(r[k] for k in KEYS[table]))

    def snapshot(self) -> Snapshot:
        dbs = ", ".join(f"'{self.db(layer)}'" for layer in TABLES)
        columns: dict[str, list[dict[str, Any]]] = {}
        for col in self.query(
            "select table_name, column_name, data_type from information_schema.columns "
            f"where table_schema in ({dbs}) order by table_name, ordinal_position"
        ):
            columns.setdefault(col["table_name"], []).append(col)
        jobs = [(layer, t) for layer, tables in TABLES.items() for t in tables]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = pool.map(lambda job: self.fetch(*job, columns[job[1]]), jobs)
        return {table: result for (_, table), result in zip(jobs, rows, strict=True)}


@pytest.fixture(scope="module")
def runs() -> dict[str, Snapshot]:
    lh = Lakehouse()
    try:
        lh.create_raw_tables()
        lh.upload("batch1", fx.BATCH1_GAMES, fx.BATCH1_REVIEWS)
        lh.dbt("run")
        snaps = {"batch1": lh.snapshot()}
        lh.dbt("run")
        snaps["rerun"] = lh.snapshot()
        lh.upload("batch2", fx.BATCH2_GAMES, fx.BATCH2_REVIEWS)
        lh.dbt("run")
        snaps["batch2"] = lh.snapshot()
        lh.dbt("build", full_refresh=True)  # + all dbt data tests
        snaps["full"] = lh.snapshot()
        yield snaps
    finally:
        if os.environ.get("ETL_CI_KEEP", "false").lower() != "true":
            lh.teardown()


def by(rows: list[dict], key: str) -> dict[Any, dict]:
    return {r[key]: r for r in rows}


def without_batch(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "_batch_at"} for r in rows]


# ---- idempotency / incrementality -----------------------------------------------------------


def test_rerun_without_new_data_changes_nothing(runs):
    for table, rows in runs["batch1"].items():
        assert runs["rerun"][table] == rows, table


def test_incremental_matches_full_refresh(runs):
    for table in ("int_game_review_counts", "game_features", "user_features", "interactions"):
        assert without_batch(runs["full"][table]) == without_batch(runs["batch2"][table]), table
    for table in MART_CONTRACTS:
        if table.startswith("lkp_"):  # never rebuilt, ids stay as assigned
            assert runs["full"][table] == runs["batch2"][table], table


@pytest.mark.parametrize("run", ["batch1", "batch2", "full"])
def test_marts_match_contracts(runs, run):
    for table, contract in MART_CONTRACTS.items():
        assert runs[run][table], f"{table} is empty"
        for row in runs[run][table]:
            contract.model_validate(row)


# ---- intermediate ----------------------------------------------------------------------------


def test_games_deduplicated(runs):
    kept = {r["game_id"] for r in runs["batch1"]["int_games__deduplicated"]}
    assert kept == {10, 21, *range(101, 108)}  # 20 loses to 21, 30 is dlc, 40 has no name
    kept2 = {r["game_id"] for r in runs["batch2"]["int_games__deduplicated"]}
    assert kept2 == kept | {50}  # 60 "ALPHA" loses to 10
    alpha = by(runs["batch2"]["int_games__deduplicated"], "game_id")[10]
    assert alpha["game_developers"] == ["Valve"]  # trimmed, blanks and duplicates dropped


def test_reviews_deduplicated_first_seen_wins(runs):
    reviews = runs["batch2"]["int_reviews__deduplicated"]
    ids = [r["review_id"] for r in reviews]
    assert len(ids) == len(set(ids))
    assert 99 not in ids
    assert {13, 14} <= set(ids)  # distinct review ids: kept here, deduplicated in the ledger
    assert by(reviews, "review_id")[2]["is_positive"] is True


def test_ledger_releases_reviews_once_their_game_is_kept(runs):
    batch1 = {r["review_id"] for r in runs["batch1"]["int_review_events"]}
    assert batch1 == {1, 2, 3, 5, 6, *range(901, 908)}
    batch2 = {r["review_id"] for r in runs["batch2"]["int_review_events"]}
    assert batch2 == batch1 | {7, 8, 10, 11}  # 7 waited for game 50; 4 and 12 never
    # one review per (user, game): 13 repeats 1 in the same batch, 14 repeats 5 across runs
    assert not {13, 14} & batch2


# ---- marts -----------------------------------------------------------------------------------


def test_lookups_are_dense_stable_and_skip_reserved_ids(runs):
    names = {t: {r["name"]: r["id"] for r in runs["batch2"][t]} for t in
             ("lkp_developers", "lkp_publishers", "lkp_genres", "lkp_categories")}  # fmt: skip
    assert names["lkp_developers"] == {"Indie Co": 2, "Studio X": 3, "Valve": 4, "New Studio": 5}
    assert names["lkp_publishers"] == {"Big Pub": 2, "Pub X": 3, "Valve": 4}
    assert names["lkp_genres"] == {"Action": 2, "Indie": 3, "RPG": 4, "Strategy": 5}
    assert names["lkp_categories"] == {"Multi-player": 2, "Single-player": 3}
    games = {r["game_id"]: r["game_idx"] for r in runs["batch2"]["lkp_games"]}
    assert games == {10: 2, 21: 3, **{g: g - 97 for g in range(101, 108)}, 50: 11}
    for table in ("lkp_developers", "lkp_games"):
        assert all(r in runs["batch2"][table] for r in runs["batch1"][table])  # append-only


def laplace(positive: int, negative: int) -> float:
    return (positive + 1) / (positive + negative + 2)  # var reviews_ratio_prior = 1


def test_review_counts_are_cumulative_per_second(runs):
    rows = [r for r in runs["batch2"]["int_game_review_counts"] if r["game_id"] == 10]
    got = [(r["timestamp"], r["positive_reviews"], r["negative_reviews"]) for r in rows]
    assert got == [(EPOCH, 0, 0), (ts(1000), 1, 0), (ts(1500), 2, 1), (ts(6000), 3, 1)]
    assert len(runs["batch2"]["int_game_review_counts"]) == 25  # 10 base rows + 15 seconds


def test_game_features_smoothed_ratio_and_encoded_attributes(runs):
    rows = [r for r in runs["batch2"]["game_features"] if r["game_id"] == 10]
    assert [r["timestamp"] for r in rows] == [EPOCH, ts(1000), ts(1500), ts(6000)]
    assert [r["game_reviews_ratio"] for r in rows] == pytest.approx(
        [laplace(0, 0), laplace(1, 0), laplace(2, 1), laplace(3, 1)]
    )
    alpha = rows[-1]
    assert (alpha["game_idx"], alpha["game_developers"], alpha["game_publishers"]) == (2, [4], [4])
    assert (alpha["game_genres"], alpha["game_categories"]) == ([2], [3, 2])
    assert len(runs["batch2"]["game_features"]) == 25


def test_user_features_last_five_positive_most_recent_first(runs):
    latest = {}
    for r in runs["batch2"]["user_features"]:
        latest[r["user_id"]] = r["games_reviewed_positive"]  # rows sorted by timestamp
    assert latest[900] == [10, 9, 8, 7, 6]
    assert latest[100] == [11, 3, 2, 0, 0]
    assert 200 in latest and 600 not in latest  # 600 only has a negative review


def test_interactions_use_features_strictly_before_the_review(runs):
    rows = by(runs["batch2"]["interactions"], "review_id")
    zero = [0, 0, 0, 0, 0]

    def feats(review_id):
        r = rows[review_id]
        return r["games_reviewed_positive"], r["game_reviews_ratio"]

    # (history, ratio from the game's counts strictly before the review)
    expected = {
        1: (zero, laplace(0, 0)),
        3: (zero, laplace(1, 0)),  # same second as 6: neither sees the other
        6: (zero, laplace(1, 0)),
        10: (zero, laplace(2, 1)),
        2: ([2, 0, 0, 0, 0], laplace(0, 0)),
        5: (zero, laplace(1, 0)),
        8: ([3, 2, 0, 0, 0], laplace(1, 0)),
        7: (zero, laplace(0, 0)),
        11: (zero, laplace(2, 0)),
    }
    for review_id, (history, ratio) in expected.items():
        got_history, got_ratio = feats(review_id)
        assert got_history == history, review_id
        assert got_ratio == pytest.approx(ratio), review_id
    assert rows[907]["games_reviewed_positive"] == [9, 8, 7, 6, 5]
    assert rows[3]["is_positive"] is False and rows[2]["is_positive"] is True
    assert set(rows) == {r["review_id"] for r in runs["batch2"]["int_review_events"]}
