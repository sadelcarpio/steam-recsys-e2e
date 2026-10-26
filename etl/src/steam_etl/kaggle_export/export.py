"""The export flow: due? -> UNLOAD -> download + consolidate -> checks -> publish -> state."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from steam_etl.kaggle_export.athena import Athena
from steam_etl.kaggle_export.publisher import KagglePublishError, Publisher
from steam_etl.kaggle_export.settings import KaggleExportSettings

log = logging.getLogger(__name__)

TABLES = ("reviews", "games", "game_tags")
REPO_URL = "https://github.com/sadelcarpio/steam-recsys-e2e"
TITLE = "Steam Reviews, Games and Tags"
SUBTITLE = "Steam user reviews with pseudonymous user ids, game details and user tags"
METADATA_FILE = "dataset-metadata.json"


class KaggleExportState(BaseModel):
    """`last.json`: the last successful publish."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    published_at: datetime
    run_id: str
    version: int
    rows: dict[str, int]


class ExportError(RuntimeError):
    pass


def sql(name: str, settings: KaggleExportSettings) -> str:
    text = resources.files(__package__).joinpath(f"sql/{name}.sql").read_text()
    return text.format(
        raw=settings.raw_database,
        intermediate=settings.intermediate_database,
        marts=settings.marts_database,
    )


def unload_sql(query: str, location: str) -> str:
    return (
        f"UNLOAD ({query.strip()})\nTO '{location}'\n"
        "WITH (format = 'PARQUET', compression = 'ZSTD')"
    )


def split_s3(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def load_state(s3: Any, uri: str) -> KaggleExportState | None:
    bucket, key = split_s3(uri)
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    return KaggleExportState.model_validate_json(body)


def save_state(s3: Any, uri: str, state: KaggleExportState) -> None:
    bucket, key = split_s3(uri)
    s3.put_object(Bucket=bucket, Key=key, Body=state.model_dump_json(indent=2).encode())


def due(state: KaggleExportState | None, settings: KaggleExportSettings, now: datetime) -> bool:
    if state is None or state.dataset != settings.kaggle_dataset:
        return True
    return now - state.published_at >= timedelta(days=settings.kaggle_min_days_between)


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [obj["Key"] for obj in page.get("Contents", []) if obj["Size"] > 0]
    return sorted(keys)


def delete_prefix(s3: Any, bucket: str, prefix: str) -> None:
    keys = list_keys(s3, bucket, prefix)
    for start in range(0, len(keys), 1000):
        chunk = [{"Key": key} for key in keys[start : start + 1000]]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})


def consolidate(
    s3: Any,
    bucket: str,
    keys: list[str],
    out_dir: Path,
    name: str,
    *,
    rows_per_file: int,
    tmp: Path,
) -> int:
    """Downloads the UNLOAD parts one at a time and rewrites them as `<name>.parquet`, or
    `<name>-NNN.parquet` when there is more than `rows_per_file` rows. Returns the row count."""
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    files: list[Path] = []
    in_file = total = 0
    for key in keys:
        part = tmp / "part.parquet"
        s3.download_file(bucket, key, str(part))
        parquet = pq.ParquetFile(part)
        schema = schema or parquet.schema_arrow
        for batch in parquet.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch]).cast(schema)
            while table.num_rows:
                if writer is None or in_file >= rows_per_file:
                    if writer is not None:
                        writer.close()
                    files.append(out_dir / f"{name}-{len(files):03d}.parquet")
                    writer = pq.ParquetWriter(files[-1], schema, compression="zstd")
                    in_file = 0
                chunk = table.slice(0, rows_per_file - in_file)
                writer.write_table(chunk)
                table = table.slice(chunk.num_rows)
                in_file += chunk.num_rows
                total += chunk.num_rows
        part.unlink()
    if writer is not None:
        writer.close()
    if len(files) == 1:
        files[0].rename(out_dir / f"{name}.parquet")
    return total


def dataset_metadata(settings: KaggleExportSettings, rows: dict[str, int], now: datetime) -> dict:
    card = Template(resources.files(__package__).joinpath("datacard.md").read_text())
    return {
        "id": settings.kaggle_dataset,
        "title": TITLE,
        "subtitle": SUBTITLE,
        "description": card.substitute(
            repo_url=REPO_URL,
            snapshot_date=now.date().isoformat(),
            reviews=f"{rows['reviews']:,}",
            games=f"{rows['games']:,}",
            tags=f"{rows['game_tags']:,}",
        ),
        "licenses": [{"name": settings.kaggle_license}],
    }


def wait_ready(
    publisher: Publisher,
    previous_version: int,
    settings: KaggleExportSettings,
    sleep: Callable[[float], None],
) -> int:
    """Waits until Kaggle has processed a version newer than `previous_version`."""
    deadline = time.monotonic() + settings.kaggle_status_timeout_minutes * 60
    while True:
        status = publisher.status()
        if status is not None and status.version > previous_version:
            if status.status == "ready":
                return status.version
            if status.status == "error":
                raise KagglePublishError(f"Kaggle failed to process version {status.version}")
        if time.monotonic() >= deadline:
            raise KagglePublishError(
                f"version not ready after {settings.kaggle_status_timeout_minutes} min"
            )
        sleep(settings.kaggle_status_poll_seconds)


def run_export(
    settings: KaggleExportSettings,
    *,
    s3: Any,
    athena: Athena,
    publisher_factory: Callable[[], Publisher],
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Returns "disabled", "not due" or "published"; raises on any failure (nothing is
    published unless every check passed)."""
    now = now or datetime.now(UTC)
    if not settings.kaggle_export_enabled:
        log.info("Kaggle export disabled (KAGGLE_EXPORT_ENABLED=false)")
        return "disabled"
    state = load_state(s3, settings.export_state_s3_uri)
    if not due(state, settings, now):
        log.info(
            "last publish %s (run %s) is newer than %d days: skipped",
            state.published_at.isoformat(),
            state.run_id,
            settings.kaggle_min_days_between,
        )
        return "not due"

    bucket, base = split_s3(settings.export_s3_prefix.rstrip("/") + f"/{settings.run_id}/")
    delete_prefix(s3, bucket, base)  # UNLOAD needs an empty destination (a retried run id)
    work = Path(settings.work_dir) / settings.run_id
    shutil.rmtree(work, ignore_errors=True)
    folder, tmp = work / "dataset", work / "tmp"
    folder.mkdir(parents=True)
    tmp.mkdir()
    try:
        rows: dict[str, int] = {}
        for table in TABLES:
            log.info("unloading %s", table)
            athena.run(unload_sql(sql(table, settings), f"s3://{bucket}/{base}{table}/"))
            keys = list_keys(s3, bucket, f"{base}{table}/")
            rows[table] = consolidate(
                s3, bucket, keys, folder, table, rows_per_file=settings.rows_per_file, tmp=tmp
            )
            log.info("%s: %d rows in %d parts", table, rows[table], len(keys))

        expected = int(athena.scalar(sql("count_reviews", settings)))
        if rows["reviews"] != expected:
            raise ExportError(
                f"reviews: {rows['reviews']} rows exported, int_reviews__deduplicated has "
                f"{expected} (a scrape or ETL run in between?)"
            )
        for table in ("games", "game_tags"):
            if rows[table] == 0:
                raise ExportError(f"{table}: no rows")

        (folder / METADATA_FILE).write_text(
            json.dumps(dataset_metadata(settings, rows, now), indent=2, ensure_ascii=False)
        )
        publisher = publisher_factory()
        current = publisher.status()
        note = f"Pipeline run {settings.run_id}: {rows['reviews']:,} reviews"
        if current is None:
            log.info("creating private dataset %s", settings.kaggle_dataset)
            publisher.create(folder)
            previous = 0
        else:
            log.info("new version of %s (current v%d)", settings.kaggle_dataset, current.version)
            publisher.version(folder, note)
            previous = current.version
        version = wait_ready(publisher, previous, settings, sleep)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        delete_prefix(s3, bucket, base)

    save_state(
        s3,
        settings.export_state_s3_uri,
        KaggleExportState(
            dataset=settings.kaggle_dataset,
            published_at=now,
            run_id=settings.run_id,
            version=version,
            rows=rows,
        ),
    )
    log.info("published %s v%d: %s", settings.kaggle_dataset, version, rows)
    return "published"
