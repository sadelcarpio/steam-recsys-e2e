import io
import json
import logging
import re
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from steam_etl.config import DEFAULT_DBT_PROJECT_DIR
from steam_etl.kaggle_export.export import (
    ExportError,
    KaggleExportState,
    dataset_metadata,
    run_export,
    save_state,
    sql,
    wait_ready,
)
from steam_etl.kaggle_export.publisher import DatasetStatus, KagglePublisher, KagglePublishError
from steam_etl.kaggle_export.settings import KaggleExportSettings

BUCKET = "processed-bucket"
STATE_URI = f"s3://{BUCKET}/exports/kaggle-state/last.json"
NOW = datetime(2026, 10, 29, 22, 0, tzinfo=UTC)
TABLE_ROWS = {"reviews": 25, "games": 4, "game_tags": 3}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("USE_SSM", raising=False)

    def make(**overrides) -> KaggleExportSettings:
        values = {
            "athena_work_group": "wg",
            "export_s3_prefix": f"s3://{BUCKET}/exports/kaggle/",
            "export_state_s3_uri": STATE_URI,
            "run_id": "exec-1",
            "kaggle_export_enabled": True,
            "kaggle_dataset": "sergiodelcarpio/steam-reviews",
            "work_dir": str(tmp_path / "work"),
            "rows_per_file": 10,
            "kaggle_status_poll_seconds": 0.01,
        }
        return KaggleExportSettings(**{**values, **overrides})

    return make


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


class FakeAthena:
    """UNLOAD writes two Parquet parts per table to (moto) S3; the count query returns the
    reviews count (or `count`)."""

    def __init__(self, s3, rows=None, count=None) -> None:
        self.s3 = s3
        self.rows = rows or TABLE_ROWS
        self.count = count
        self.statements: list[str] = []

    def run(self, statement: str) -> str:
        self.statements.append(statement)
        location = re.search(r"TO 's3://([^/]+)/(.+)'", statement)
        bucket, prefix = location.groups()
        table = prefix.rstrip("/").rsplit("/", 1)[-1]
        n = self.rows[table]
        for part, (start, stop) in enumerate([(0, n // 2), (n // 2, n)]):
            buf = io.BytesIO()
            pq.write_table(pa.table({"id": list(range(start, stop))}), buf)
            self.s3.put_object(Bucket=bucket, Key=f"{prefix}part-{part}", Body=buf.getvalue())
        return "query-id"

    def scalar(self, statement: str) -> str:
        self.statements.append(statement)
        return str(self.rows["reviews"] if self.count is None else self.count)


class FakePublisher:
    def __init__(self, existing: int | None = None, final: str = "ready") -> None:
        self.version_number = existing
        self.final = final
        self.calls: list[tuple] = []
        self.files: list[str] = []
        self.metadata: dict = {}

    def status(self):
        if self.version_number is None:
            return None
        return DatasetStatus(self.final, self.version_number)

    def _publish(self, folder: Path):
        self.files = sorted(p.name for p in folder.iterdir())
        self.metadata = json.loads((folder / "dataset-metadata.json").read_text())
        self.version_number = (self.version_number or 0) + 1

    def create(self, folder: Path) -> None:
        self.calls.append(("create",))
        self._publish(folder)

    def version(self, folder: Path, note: str) -> None:
        self.calls.append(("version", note))
        self._publish(folder)


def export(settings, s3, athena=None, publisher=None, now=NOW):
    athena = athena or FakeAthena(s3)
    publisher = publisher or FakePublisher()
    outcome = run_export(
        settings,
        s3=s3,
        athena=athena,
        publisher_factory=lambda: publisher,
        now=now,
        sleep=lambda _s: None,
    )
    return outcome, athena, publisher


def stored_state(s3) -> KaggleExportState:
    body = s3.get_object(Bucket=BUCKET, Key="exports/kaggle-state/last.json")["Body"].read()
    return KaggleExportState.model_validate_json(body)


def test_first_publish_creates_a_private_dataset(settings, s3):
    s = settings()
    outcome, athena, publisher = export(s, s3)

    assert outcome == "published"
    assert publisher.calls == [("create",)]
    # 25 reviews at 10 rows per file -> 3 files; small tables keep a single file
    assert publisher.files == [
        "dataset-metadata.json",
        "game_tags.parquet",
        "games.parquet",
        "reviews-000.parquet",
        "reviews-001.parquet",
        "reviews-002.parquet",
    ]
    assert publisher.metadata["id"] == "sergiodelcarpio/steam-reviews"
    assert "| 25 | review |" in publisher.metadata["description"]
    assert "2026-10-29" in publisher.metadata["description"]
    unloads = [q for q in athena.statements if q.startswith("UNLOAD")]
    assert len(unloads) == 3
    assert all("exports/kaggle/exec-1/" in q and "format = 'PARQUET'" in q for q in unloads)

    state = stored_state(s3)
    assert (state.version, state.run_id, state.rows) == (1, "exec-1", TABLE_ROWS)
    # the unloaded parts and the local copy are removed
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix="exports/kaggle/").get("KeyCount") == 0
    assert not (Path(s.work_dir) / "exec-1").exists()


def test_existing_dataset_gets_a_new_version(settings, s3):
    _, _, publisher = export(settings(), s3, publisher=FakePublisher(existing=7))
    assert publisher.calls == [("version", "Pipeline run exec-1: 25 reviews")]
    assert stored_state(s3).version == 8


def test_disabled_does_nothing(settings, s3):
    outcome, athena, publisher = export(settings(kaggle_export_enabled=False), s3)
    assert outcome == "disabled"
    assert athena.statements == [] and publisher.calls == []


def test_frequency(settings, s3):
    s = settings()
    save_state(
        s3,
        STATE_URI,
        KaggleExportState(
            dataset=s.kaggle_dataset,
            published_at=NOW - timedelta(days=27),
            run_id="old",
            version=3,
            rows=TABLE_ROWS,
        ),
    )
    outcome, athena, _ = export(s, s3)
    assert outcome == "not due" and athena.statements == []

    assert export(s, s3, now=NOW + timedelta(days=1))[0] == "published"
    # every run when the minimum is 0, and always for another dataset
    later = NOW + timedelta(days=1)
    assert export(settings(kaggle_min_days_between=0, run_id="r2"), s3, now=later)[0] == "published"
    other = settings(kaggle_dataset="sergiodelcarpio/other", run_id="r3")
    assert export(other, s3, now=later)[0] == "published"


def test_count_mismatch_fails_before_any_upload(settings, s3):
    def no_publisher():
        raise AssertionError("must not reach Kaggle")

    with pytest.raises(ExportError, match="int_reviews__deduplicated has 24"):
        run_export(
            settings(),
            s3=s3,
            athena=FakeAthena(s3, count=24),
            publisher_factory=no_publisher,
            now=NOW,
        )
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix="exports/").get("KeyCount") == 0


def test_empty_games_fail(settings, s3):
    athena = FakeAthena(s3, rows={**TABLE_ROWS, "games": 0})
    with pytest.raises(ExportError, match="games: no rows"):
        export(settings(), s3, athena=athena)


def test_kaggle_processing_error_keeps_the_previous_state(settings, s3):
    with pytest.raises(KagglePublishError, match="failed to process"):
        export(settings(), s3, publisher=FakePublisher(existing=2, final="error"))
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix="exports/kaggle-state/").get("KeyCount") == 0


def test_wait_ready_times_out(settings):
    class Pending:
        def status(self):
            return DatasetStatus("pending", 2)

    s = settings(kaggle_status_timeout_minutes=1)
    clock = iter(range(0, 10_000, 30))
    import steam_etl.kaggle_export.export as module

    original = module.time.monotonic
    module.time.monotonic = lambda: float(next(clock))
    try:
        with pytest.raises(KagglePublishError, match="not ready"):
            wait_ready(Pending(), 1, s, sleep=lambda _s: None)
    finally:
        module.time.monotonic = original


# ---- SQL ---------------------------------------------------------------------------------------

PUBLISHED_REVIEW_COLUMNS = [
    "review_id", "user_id", "game_id", "is_positive", "review", "language", "reviewed_at",
    "updated_at", "playtime_forever", "playtime_last_two_weeks", "playtime_at_review",
    "num_games_owned", "num_reviews", "votes_up", "votes_funny", "weighted_vote_score",
    "comment_count", "steam_purchase", "received_for_free", "written_during_early_access",
    "primarily_steam_deck",
]  # fmt: skip


def outer_select(query: str) -> list[str]:
    body = re.sub(r"--.*", "", query)
    columns = re.search(r"^select\s+(.*?)\s+from\s*\(", body, re.S | re.M).group(1)
    return [c.strip() for c in columns.split(",")]


def test_reviews_query_publishes_exactly_the_documented_columns(settings):
    query = sql("reviews", settings())
    assert outer_select(query) == PUBLISHED_REVIEW_COLUMNS
    assert "from steam_raw.reviews" in query
    for never in ("scrape_date", "last_played", "author_id", "user_idx"):
        assert never not in PUBLISHED_REVIEW_COLUMNS


def test_reviews_dedup_rule_matches_the_intermediate_model(settings):
    def clause(text: str) -> str:
        match = re.search(r"row_number\(\)\s+over\s*\((.*?)\)", text, re.S)
        return " ".join(match.group(1).split())

    model = (
        DEFAULT_DBT_PROJECT_DIR / "models/intermediate/int_reviews__deduplicated.sql"
    ).read_text()
    assert clause(sql("reviews", settings())) == clause(model)


def test_queries_resolve_their_databases(settings):
    s = settings()
    assert "steam_marts.game_details" in sql("games", s)
    assert "game_name_key" not in re.sub(r"--.*", "", sql("games", s))
    assert "steam_raw.game_tags" in sql("game_tags", s)
    assert sql("count_reviews", s).strip().endswith("steam_intermediate.int_reviews__deduplicated")
    for name in ("reviews", "games", "game_tags", "count_reviews"):
        assert "{" not in sql(name, s)


def test_metadata_uses_the_license_setting(settings):
    meta = dataset_metadata(settings(kaggle_license="CC0-1.0"), TABLE_ROWS, NOW)
    assert meta["licenses"] == [{"name": "CC0-1.0"}]
    assert 20 <= len(meta["subtitle"]) <= 80
    assert "not** a Steam id" in meta["description"]


def test_default_license_claims_no_rights_over_the_content(settings):
    meta = dataset_metadata(settings(), TABLE_ROWS, NOW)
    assert meta["licenses"] == [{"name": "copyright-authors"}]
    assert "Original Authors" in meta["description"]
    assert "steam-recsys-e2e/issues" in meta["description"]


# ---- the token -------------------------------------------------------------------------------


def fake_publisher(monkeypatch, tmp_path, api, token="t"):
    module = types.ModuleType("kaggle.api.kaggle_api_extended")
    module.KaggleApi = api
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)
    # set first, so monkeypatch removes what the publisher writes into os.environ
    monkeypatch.setenv("KAGGLE_API_TOKEN", "placeholder")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    return KagglePublisher("o/d", token, config_dir=str(tmp_path))


class StatusApi:
    """Owns `o/d-old` and, when `exists`, `o/d`; any status call of a missing dataset fails."""

    exists = False

    def authenticate(self):
        pass

    def dataset_list(self, mine=False, search=None):
        assert mine and search == "d"
        refs = ["o/d-old"] + (["o/d"] if self.exists else [])
        return [types.SimpleNamespace(ref=ref) for ref in refs]

    def dataset_status(self, dataset, format=None):
        if not self.exists:
            raise RuntimeError("Kaggle answers a never-created dataset with a 403")
        return json.dumps({"status": "ready", "current_version_number": 3})


def test_status_of_a_dataset_not_created_yet_is_none(monkeypatch, tmp_path):
    assert fake_publisher(monkeypatch, tmp_path, StatusApi).status() is None


def test_status_of_an_existing_dataset(monkeypatch, tmp_path):
    api = type("ExistingApi", (StatusApi,), {"exists": True})
    assert fake_publisher(monkeypatch, tmp_path, api).status() == DatasetStatus("ready", 3)


def test_publisher_errors_carry_the_http_codes_only(monkeypatch, tmp_path):
    class Response:
        status_code = 200

        def json(self):
            return {"code": 403, "message": "Permission denied for user o"}

    class HTTPError(Exception):
        response = Response()

    class FailingApi(StatusApi):
        def dataset_list(self, mine=False, search=None):
            raise HTTPError("Permission denied for user o")

    with pytest.raises(KagglePublishError) as err:
        fake_publisher(monkeypatch, tmp_path, FailingApi).status()
    assert str(err.value) == "list of o/d failed: HTTPError (HTTP 200, body 403)"


def test_publisher_never_surfaces_the_token(monkeypatch, tmp_path, caplog):
    token = "KGAT_secret_token_value"

    class FakeApi:
        def authenticate(self):
            pass

        def dataset_list(self, mine=False, search=None):
            raise RuntimeError(f"401 for request with Authorization: Bearer {token}")

    caplog.set_level(logging.DEBUG)
    publisher = fake_publisher(monkeypatch, tmp_path, FakeApi, token=token)
    with pytest.raises(KagglePublishError) as err:
        publisher.status()
    assert token not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert token not in caplog.text
