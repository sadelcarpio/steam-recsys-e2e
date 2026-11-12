from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from steam_eda import cached, files, load_games, load_reviews, parse_release_date, user_sample


def test_files_match_each_table_only(dataset) -> None:
    assert [p.rsplit("/", 1)[-1] for p in files("reviews", str(dataset))] == [
        "reviews-000.parquet",
        "reviews-001.parquet",
    ]
    assert files("games", str(dataset))[0].endswith("/games.parquet")
    assert files("game_tags", str(dataset))[0].endswith("/game_tags.parquet")


def test_files_reads_unload_part_folders(tmp_path) -> None:
    (tmp_path / "reviews").mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "reviews" / "part-0001.parquet")
    assert files("reviews", str(tmp_path))[0].endswith("reviews/part-0001.parquet")
    with pytest.raises(FileNotFoundError):
        files("games", str(tmp_path))
    with pytest.raises(ValueError, match="unknown table"):
        files("users", str(tmp_path))


def test_load_reviews_skips_text_and_samples_whole_users(env) -> None:
    every = load_reviews(fraction=1)
    assert "review" not in every.columns
    half = load_reviews(fraction=0.5)
    assert 0 < half["user_id"].n_unique() < every["user_id"].n_unique()
    kept = set(half["user_id"])
    # whole histories: every review of a kept user is kept
    assert every.filter(pl.col("user_id").is_in(kept)).height == half.height
    assert set(load_reviews(fraction=0.5)["user_id"]) == kept


def test_user_sample_is_deterministic() -> None:
    df = pl.DataFrame({"user_id": list(range(10_000))})
    a = df.filter(user_sample(0.1, seed=1))
    assert 800 < a.height < 1200
    assert a.equals(df.filter(user_sample(0.1, seed=1)))
    assert not a.equals(df.filter(user_sample(0.1, seed=2)))


def test_parse_release_date() -> None:
    s = pl.Series(["Oct 9, 2020", "9 Oct, 2020", "October 9, 2020", " Aug 21, 2012 ",
                   "Q3 2026", "2026", "Coming soon", None])  # fmt: skip
    parsed = pl.select(parse_release_date(pl.lit(s))).to_series().to_list()
    assert parsed[:4] == [date(2020, 10, 9)] * 3 + [date(2012, 8, 21)]
    assert parsed[4:] == [None] * 4


def test_load_games_joins_tags_and_dates(env) -> None:
    games = load_games()
    assert games["game_id"].is_unique().all()
    assert {"tag_names", "release_date", "release_year"} <= set(games.columns)
    released = games.filter(~pl.col("game_coming_soon"))
    assert released["release_date"].null_count() == 0
    assert games.filter(pl.col("game_coming_soon"))["release_date"].null_count() > 0


def test_cached_builds_once(env) -> None:
    calls = []

    def build():
        calls.append(1)
        return pl.DataFrame({"a": [1]})

    assert cached("x", build).equals(cached("x", build))
    assert len(calls) == 1
    cached("x", build, refresh=True)
    assert len(calls) == 2
