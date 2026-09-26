"""Game search index for the frontend: every game of the online catalog (the games that
POST /recommendations can use) as [appid, name, reviews], gzipped JSON. The browser downloads
it once through CloudFront (/data/games.json) and searches it locally.

Written with the online catalog (same run, same games), to
s3://<model-artifacts>/<SEARCH_INDEX_KEY> with `Content-Encoding: gzip`, so browsers decompress
it transparently. `index_from_catalog` rebuilds it from a published catalog.npz without a run
(scripts/publish_search_index.py).
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping
from datetime import datetime

import numpy as np

from steam_inference.contracts import SearchIndex
from steam_inference.features import InferenceData


def search_index(
    game_ids: np.ndarray,
    names: list[str],
    reviews: np.ndarray,
    *,
    model_id: str,
    now: datetime,
) -> SearchIndex:
    """Most reviewed first (ties: lowest appid); games without a name are left out."""
    order = np.lexsort((game_ids, -reviews))
    return SearchIndex(
        model_id=model_id,
        generated_at=now,
        games=[
            (int(game_ids[row]), names[row], int(reviews[row]))
            for row in order
            if names[row].strip()
        ],
    )


def build_search_index(
    data: InferenceData,
    *,
    model_id: str,
    now: datetime,
    excluded: np.ndarray | None = None,
) -> SearchIndex:
    """The run's catalog, minus the `excluded` rows (bool per row). `reviews` counts the
    reviews loaded by this run (every user's when MAX_USERS=0), not Steam's total."""
    games = data.games
    reviewed = data.reviews.game_idx[data.reviews.game_idx >= 0]
    per_idx = np.bincount(reviewed, minlength=len(games.catalog.row_of))
    game_idx = games.catalog.items.game_idx.astype(np.int64)
    reviews = per_idx[np.clip(game_idx, 0, len(per_idx) - 1)] * (game_idx < len(per_idx))
    rows = np.flatnonzero(~excluded) if excluded is not None else np.arange(len(games))
    return search_index(
        games.game_id[rows],
        [str(n) for n in games.name[rows]],
        reviews[rows],
        model_id=model_id,
        now=now,
    )


def index_from_catalog(
    arrays: Mapping[str, np.ndarray],
    reviews_by_game_id: Mapping[int, int],
    *,
    model_id: str,
    now: datetime,
) -> SearchIndex:
    """A published online catalog (`contracts.ONLINE_BUNDLE_ARRAYS`), with review counts per
    appid from elsewhere (games missing from it count 0)."""
    game_ids = arrays["item_game_id"].astype(np.int64)
    utf8, offsets = arrays["item_name_utf8"].tobytes(), arrays["item_name_offsets"]
    names = [utf8[offsets[i] : offsets[i + 1]].decode() for i in range(len(game_ids))]
    reviews = np.array([reviews_by_game_id.get(int(g), 0) for g in game_ids], dtype=np.int64)
    return search_index(game_ids, names, reviews, model_id=model_id, now=now)


def encode_search_index(index: SearchIndex) -> bytes:
    return gzip.compress(index.model_dump_json().encode(), compresslevel=9, mtime=0)
