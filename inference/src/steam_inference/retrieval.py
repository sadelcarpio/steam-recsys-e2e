"""Stage 2: exact two-tower retrieval (no ANN).

Every game of the current catalog is embedded once by the item tower; users are scored in chunks
against all of them (one matmul per chunk), their reviewed games are masked out and the top K
catalog rows are kept. Brute force is cheap at this size (~1.4M users x ~50k games x 64 dims
is a few TFLOP, minutes on a Fargate CPU) and exact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from steam_training.evaluation import embed_catalog
from steam_training.model import TwoTowerModel

from steam_inference.features import Games, Reviews, Users

log = logging.getLogger(__name__)
NO_ROW = -1


@dataclass(frozen=True)
class Candidates:
    """Top K catalog rows per user, best first. Users who reviewed almost the whole catalog
    have fewer than K candidates: their tail is NO_ROW."""

    rows: np.ndarray  # int64 [users, K] catalog rows, NO_ROW = none
    scores: np.ndarray  # float32 [users, K] cosine similarity
    # float32 [catalog rows, output_dim]: the L2-normalized item embeddings scored against
    # (also exported for online serving, `online.build_catalog`)
    item_embeddings: np.ndarray

    def of(self, user: int) -> tuple[np.ndarray, np.ndarray]:
        valid = self.rows[user] != NO_ROW
        return self.rows[user][valid], self.scores[user][valid]


@torch.no_grad()
def retrieve(
    model: TwoTowerModel,
    games: Games,
    users: Users,
    reviews: Reviews,
    *,
    k: int,
    user_batch_size: int,
    item_batch_size: int,
) -> Candidates:
    model.eval()
    started = time.monotonic()
    item_embeddings = embed_catalog(model, games.catalog, item_batch_size)
    k = min(k, len(games))
    rows = np.full((len(users), k), NO_ROW, dtype=np.int64)
    scores = np.zeros((len(users), k), dtype=np.float32)
    for start in range(0, len(users), user_batch_size):
        chunk = users.take(np.arange(start, min(start + user_batch_size, len(users))))
        user_embeddings = model.user_tower(torch.from_numpy(chunk.history))
        chunk_scores = user_embeddings @ item_embeddings.T
        # exclude every game the user already reviewed (positively or not)
        reviewed = reviews.of(chunk)
        catalog_rows = torch.from_numpy(games.catalog.rows(reviewed.values))
        owner = torch.from_numpy(np.repeat(np.arange(len(chunk)), np.diff(reviewed.offsets)))
        known = catalog_rows >= 0
        chunk_scores[owner[known], catalog_rows[known]] = float("-inf")
        top = torch.topk(chunk_scores, k, dim=1)
        top_rows = top.indices.numpy()
        top_scores = top.values.numpy()
        valid = np.isfinite(top_scores)
        rows[start : start + len(chunk)] = np.where(valid, top_rows, NO_ROW)
        scores[start : start + len(chunk)] = np.where(valid, top_scores, 0.0)
    log.info(
        "retrieved top %d of %d games for %d users in %.0fs",
        k,
        len(games),
        len(users),
        time.monotonic() - started,
    )
    return Candidates(rows=rows, scores=scores, item_embeddings=item_embeddings.numpy())
