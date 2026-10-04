from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

from steam_eda import Sample, collect, load_raw, raw_files, settings


def _batches(n: int = 1000, size: int = 100) -> list[pa.RecordBatch]:
    return [
        pa.RecordBatch.from_pydict(
            {
                "user_id": [i // 4 for i in range(start, start + size)],
                "is_positive": [i % 3 != 0 for i in range(start, start + size)],
            }
        )
        for start in range(0, n, size)
    ]


def test_collect_everything() -> None:
    df = collect(_batches())
    assert df.height == 1000
    assert df.schema == pl.Schema({"user_id": pl.Int64, "is_positive": pl.Boolean})


def test_collect_filter_and_limit_stop_early() -> None:
    consumed = 0

    def counting():
        nonlocal consumed
        for batch in _batches():
            consumed += 1
            yield batch

    df = collect(counting(), filter=pl.col("is_positive"), limit=150)
    assert df.height == 150 and df["is_positive"].all()
    assert consumed == 3  # 66-67 positives per batch of 100


def test_collect_empty_keeps_schema() -> None:
    df = collect(_batches(), filter=pl.col("user_id") < 0)
    assert df.height == 0 and df.columns == ["user_id", "is_positive"]


def test_key_sample_keeps_whole_users_and_is_deterministic() -> None:
    sample = Sample.by_key("user_id", 0.3, seed=7)
    a = collect(_batches(), sample=sample)
    b = collect(_batches(size=40), sample=sample)  # batch boundaries do not matter
    assert a.equals(b)
    assert 0.15 < a.height / 1000 < 0.45
    assert (a.group_by("user_id").len()["len"] == 4).all()  # every row of a kept user


def test_key_sample_is_consistent_across_tables() -> None:
    sample = Sample.by_key("user_id", 0.2)
    users = pl.DataFrame({"user_id": list(range(250))})
    kept = set(sample.apply(users)["user_id"])
    reviews = collect(_batches(), sample=sample)
    assert set(reviews["user_id"]) == kept


def test_row_sample_is_independent_of_batching() -> None:
    sample = Sample.rows(0.25, seed=1)
    a = collect(_batches(size=100), sample=sample)
    b = collect(_batches(size=25), sample=sample)
    assert a.equals(b)
    assert 0.15 < a.height / 1000 < 0.35


def test_sample_key_must_be_loaded() -> None:
    with pytest.raises(ValueError, match="not a loaded column"):
        collect(_batches(), sample=Sample.by_key("author_id", 0.5))


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    for day in (1, 2, 3):
        pl.DataFrame(
            {
                "rec_id": [day * 10 + i for i in range(10)],
                "author_id": list(range(10)),
                "voted_up": [i % 2 == 0 for i in range(10)],
                "review": ["long text"] * 10,
            }
        ).write_parquet(reviews / f"2026-09-0{day}-000-0000.parquet")
    monkeypatch.setenv("RAW_ROOT", str(tmp_path))
    settings.cache_clear()
    yield tmp_path
    settings.cache_clear()


def test_raw_files_newest_last(raw_dir) -> None:
    files = raw_files("reviews")
    assert [f.rsplit("/", 1)[1][:10] for f in files] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert raw_files("reviews", last=1) == files[-1:]
    with pytest.raises(ValueError):
        raw_files("users")


def test_load_raw_skips_texts_and_samples_by_key(raw_dir) -> None:
    df = load_raw("reviews")
    assert df.height == 30 and "review" not in df.columns
    sample = Sample.by_key("author_id", 0.5, seed=3)
    sampled = load_raw("reviews", sample=sample, filter=pl.col("voted_up"))
    kept = set(sample.apply(pl.DataFrame({"author_id": list(range(10))}))["author_id"])
    assert set(sampled["author_id"]) == kept & {0, 2, 4, 6, 8}
    assert load_raw("reviews", columns=["rec_id"], last_files=1, limit=3).height == 3


def test_cached_builds_once(tmp_path, monkeypatch) -> None:
    from steam_eda import cached

    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    settings.cache_clear()
    calls = []

    def build() -> pl.DataFrame:
        calls.append(1)
        return pl.DataFrame({"a": [1, 2]})

    assert cached("x", build).equals(cached("x", build))
    assert len(calls) == 1
    cached("x", build, refresh=True)
    assert len(calls) == 2
    settings.cache_clear()
