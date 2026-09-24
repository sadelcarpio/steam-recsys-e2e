"""Raw parquet fixtures for the Athena integration test, in the scraper output format
(data_ingestion/src/steam_ingestion/schemas.py), plus the Hive DDL registering them.

Batch 1 (scrape 2026-01-01) exercises: appid retry duplicates, name dedup (20 vs 21 "beta"),
non-game types, missing names, messy arrays, duplicate review ids, reviews of dropped / not yet
scraped games, invalid reviews, a double-posted review (13: same user and game as 1, 10 s
later, new review id), and a user with more than 5 positive reviews (900).
Batch 2 (scrape 2026-01-08) adds game 50 (its review 7 was scraped in batch 1), a losing
duplicate name (60 "ALPHA"), new vocabulary values, a re-scraped, edited review 2, and a
second review of an already reviewed game in a later run (14: user 200, game 21).
"""

from __future__ import annotations

from datetime import date

import pyarrow as pa

T0 = 1_700_000_000
BATCH1_DATE = date(2026, 1, 1)
BATCH2_DATE = date(2026, 1, 8)

_STR_LIST = pa.list_(pa.string())

GAMES_SCHEMA = pa.schema(
    [
        ("appid", pa.int64()),
        ("name", pa.string()),
        ("type", pa.string()),
        ("required_age", pa.int64()),
        ("is_free", pa.bool_()),
        ("minimum_pc_requirements", pa.string()),
        ("recommended_pc_requirements", pa.string()),
        ("controller_support", pa.string()),
        ("detailed_description", pa.string()),
        ("about_the_game", pa.string()),
        ("short_description", pa.string()),
        ("supported_languages", _STR_LIST),
        ("header_image", pa.string()),
        ("developers", _STR_LIST),
        ("publishers", _STR_LIST),
        ("price", pa.float64()),
        ("categories", _STR_LIST),
        ("genres", _STR_LIST),
        ("windows_support", pa.bool_()),
        ("mac_support", pa.bool_()),
        ("linux_support", pa.bool_()),
        ("release_date", pa.string()),
        ("coming_soon", pa.bool_()),
        ("recommendations", pa.int64()),
        ("dlc", pa.list_(pa.int64())),
        ("review_score", pa.int64()),
        ("review_score_desc", pa.string()),
        ("scrape_date", pa.date32()),
    ]
)

REVIEWS_SCHEMA = pa.schema(
    [
        ("rec_id", pa.int64()),
        ("author_id", pa.int64()),
        ("appid", pa.int64()),
        ("playtime_forever", pa.int64()),
        ("playtime_last_two_weeks", pa.int64()),
        ("playtime_at_review", pa.int64()),
        ("num_games_owned", pa.int64()),
        ("num_reviews", pa.int64()),
        ("last_played", pa.int64()),
        ("language", pa.string()),
        ("review", pa.string()),
        ("timestamp_created", pa.int64()),
        ("timestamp_updated", pa.int64()),
        ("voted_up", pa.bool_()),
        ("votes_up", pa.int64()),
        ("votes_funny", pa.int64()),
        ("weighted_vote_score", pa.float64()),
        ("comment_count", pa.int64()),
        ("steam_purchase", pa.bool_()),
        ("received_for_free", pa.bool_()),
        ("written_during_early_access", pa.bool_()),
        ("primarily_steam_deck", pa.bool_()),
        ("scrape_date", pa.date32()),
    ]
)

RAW_SCHEMAS = {"games": GAMES_SCHEMA, "reviews": REVIEWS_SCHEMA}


def _game(appid, name, *, scrape_date, type_="game", devs=(), pubs=(), genres=(), cats=(),
          score=None, recs=None, is_free=False):  # fmt: skip
    return {
        "appid": appid,
        "name": name,
        "type": type_,
        "is_free": is_free,
        "developers": list(devs),
        "publishers": list(pubs),
        "genres": list(genres),
        "categories": None if cats is None else list(cats),
        "review_score": score,
        "recommendations": recs,
        "scrape_date": scrape_date,
    }


def _review(rec_id, user, appid, positive, dt, *, scrape_date, updated=None):
    return {
        "rec_id": rec_id,
        "author_id": user,
        "appid": appid,
        "voted_up": positive,
        "timestamp_created": T0 + dt,
        "timestamp_updated": T0 + (updated if updated is not None else dt),
        "language": "english",
        "scrape_date": scrape_date,
    }


_d1 = {"scrape_date": BATCH1_DATE}
_d2 = {"scrape_date": BATCH2_DATE}
_alpha = {"devs": ["Valve", " Valve ", ""], "pubs": ["Valve"], "genres": ["Action"],
          "cats": ["Single-player", "Multi-player"], "score": 8, "recs": 1000}  # fmt: skip

BATCH1_GAMES = [
    _game(10, "Alpha", **_alpha, **_d1),
    _game(10, "Alpha", **_alpha, **_d1),  # retried scraper task
    _game(
        20,
        "Beta",
        devs=["Indie Co"],
        genres=["Indie", "Action"],
        cats=None,
        score=5,
        recs=50,
        **_d1,
    ),  # fmt: skip
    _game(
        21,
        "beta",
        devs=["Indie Co"],
        pubs=["Big Pub"],
        genres=["Indie"],
        cats=["Single-player"],
        score=7,
        recs=10,
        **_d1,
    ),  # fmt: skip
    _game(30, "Gamma Soundtrack", type_="dlc", devs=["Valve"], score=9, **_d1),
    _game(40, None, devs=["Nobody"], **_d1),
    *[
        _game(g, f"Game {g}", devs=["Studio X"], pubs=["Pub X"], genres=["RPG"], score=6, **_d1)
        for g in range(101, 108)
    ],
]

BATCH1_REVIEWS = [
    _review(1, 100, 10, True, 1000, **_d1),
    _review(1, 100, 10, True, 1000, **_d1),  # re-emitted after a flush split
    _review(13, 100, 10, True, 1010, **_d1),  # double post: new id, same user and game as 1
    _review(2, 100, 21, True, 2000, **_d1),
    _review(3, 200, 10, False, 1500, **_d1),
    _review(4, 100, 20, True, 2500, **_d1),  # game 20 loses the name dedup
    _review(5, 200, 21, True, 3000, **_d1),
    _review(6, 300, 10, True, 1500, **_d1),  # same second as review 3
    _review(7, 400, 50, True, 2000, **_d1),  # game 50 is only scraped in batch 2
    _review(99, 999, 10, None, 1200, **_d1),  # no vote: dropped in staging
    *[_review(900 + k, 900, 100 + k, True, 100 * k, **_d1) for k in range(1, 8)],
]

BATCH2_GAMES = [
    _game(
        50,
        "Delta",
        devs=["Valve", "New Studio"],
        genres=["Strategy"],
        cats=["Single-player"],
        score=6,
        **_d2,
    ),  # fmt: skip
    _game(60, "ALPHA", devs=["Other"], score=3, **_d2),  # loses to 10
]

BATCH2_REVIEWS = [
    _review(8, 100, 50, True, 5000, **_d2),
    _review(10, 500, 10, True, 6000, **_d2),
    _review(11, 600, 21, False, 6000, **_d2),
    _review(12, 100, 60, True, 5500, **_d2),  # dropped game
    _review(14, 200, 21, False, 7000, **_d2),  # user 200 already reviewed game 21 (review 5)
    _review(2, 100, 21, False, 2000, updated=5800, **_d2),  # edited later: first seen wins
]


def to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


_HIVE_TYPES = {
    pa.int64(): "bigint",
    pa.string(): "string",
    pa.bool_(): "boolean",
    pa.float64(): "double",
    pa.date32(): "date",
    _STR_LIST: "array<string>",
    pa.list_(pa.int64()): "array<bigint>",
}


def external_table_ddl(database: str, table: str, location: str) -> str:
    columns = ",\n  ".join(f"`{f.name}` {_HIVE_TYPES[f.type]}" for f in RAW_SCHEMAS[table])
    return (
        f"create external table `{database}`.`{table}` (\n  {columns}\n)\n"
        f"stored as parquet location '{location}'"
    )
