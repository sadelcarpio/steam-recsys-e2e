"""Offline metrics beyond recall@K (spec 18): per-user hit rate, recall by user-activity
percentile, catalog coverage, exposure concentration and embedding-collapse checks.

Everything is computed from one scoring pass per model (`evaluation.rank_rows`): the target ranks
and what each row retrieved.
"""

from __future__ import annotations

import numpy as np
import torch

from steam_training.contracts import ActivityBucket, EmbeddingHealth, RetrievalDiagnostics
from steam_training.data import EvalRows
from steam_training.evaluation import Exposure, recall_from_ranks

# Activity percentiles of the validation users (users with at least one review before the
# cutoff; new users are their own bucket).
ACTIVITY_PERCENTILES = (50, 80, 95, 99)


def user_hit_rate(
    ranks: np.ndarray, user_id: np.ndarray, ks: list[int]
) -> tuple[dict[int, float], dict[int, float], int]:
    """(share of users with a hit in their top K, mean of each user's recall, users)."""
    if len(ranks) == 0:
        return dict.fromkeys(ks, 0.0), dict.fromkeys(ks, 0.0), 0
    _, inverse, rows_per_user = np.unique(user_id, return_inverse=True, return_counts=True)
    hit_rate, recall = {}, {}
    for k in sorted(ks):
        hits = np.bincount(inverse, weights=ranks < k, minlength=len(rows_per_user))
        hit_rate[k] = float((hits > 0).mean())
        recall[k] = float((hits / rows_per_user).mean())
    return hit_rate, recall, len(rows_per_user)


def activity_buckets(
    ranks: np.ndarray,
    user_id: np.ndarray,
    activity: np.ndarray,
    ks: list[int],
    percentiles: tuple[int, ...] = ACTIVITY_PERCENTILES,
) -> list[ActivityBucket]:
    """Recall per activity bucket: `new` (no review before the cutoff), then the percentiles of
    the returning validation users' activity (`p0-50`, `p50-80`, ...). Empty buckets (ties) are
    left out."""
    _, first = np.unique(user_id, return_index=True)
    user_activity = activity[first]
    returning = user_activity[user_activity > 0]
    edges = (
        np.percentile(returning, percentiles, method="lower").astype(np.int64)
        if len(returning)
        else np.array([], dtype=np.int64)
    )
    bounds = [0, *percentiles, 100]
    # bucket 0 = new users; bucket i >= 1 = activity in (edges[i-2], edges[i-1]]
    bucket = np.where(activity == 0, 0, 1 + np.searchsorted(edges, activity, side="left"))
    out = []
    for i in range(len(bounds)):
        mask = bucket == i
        if not mask.any():
            continue
        name = "new" if i == 0 else f"p{bounds[i - 1]}-{bounds[i]}"
        selected = ranks[mask]
        out.append(
            ActivityBucket(
                name=name,
                min_activity=int(activity[mask].min()),
                max_activity=int(activity[mask].max()),
                rows=int(mask.sum()),
                users=len(np.unique(user_id[mask])),
                recall={k: float((selected < k).mean()) for k in sorted(ks)},
            )
        )
    return out


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal, ->1 = all on one)."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64))
    n = len(ordered)
    cumulative = np.cumsum(ordered)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


def retrieval_diagnostics(
    ranks: torch.Tensor, exposure: Exposure, rows: EvalRows, ks: list[int]
) -> RetrievalDiagnostics:
    ranks_np = ranks.numpy()
    user_id = rows.user_id if rows.user_id is not None else np.arange(len(rows))
    activity = rows.activity if rows.activity is not None else np.zeros(len(rows), np.int64)
    hit_rate, user_recall, users = user_hit_rate(ranks_np, user_id, ks)
    best_rank = exposure.best_rank.cpu().numpy()
    return RetrievalDiagnostics(
        recall=recall_from_ranks(ranks, torch.from_numpy(rows.is_warm), ks),
        user_hit_rate=hit_rate,
        user_recall=user_recall,
        users=users,
        by_activity=activity_buckets(ranks_np, user_id, activity, ks) if len(rows) else [],
        coverage={k: float((best_rank < k).mean()) for k in sorted(ks)},
        exposure_gini=min(1.0, max(0.0, gini(exposure.count.cpu().numpy()))),
    )


@torch.no_grad()
def embedding_health(
    embeddings: torch.Tensor, max_rows: int = 10_000, seed: int = 0
) -> EmbeddingHealth:
    """Mean pairwise cosine and effective rank (exp of the entropy of the normalized singular
    values of the centered matrix) on up to `max_rows` random rows."""
    x = embeddings.detach().float().cpu()
    n, dim = x.shape
    if n > max_rows:
        index = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_rows]
        x = x[index]
    if len(x) < 2:
        return EmbeddingHealth(
            rows=len(x), dim=dim, mean_cosine=0.0, effective_rank=0.0, effective_rank_ratio=0.0
        )
    unit = torch.nn.functional.normalize(x, dim=1)
    total = unit.sum(dim=0)
    # mean over ordered pairs i != j of cos(i, j) = (|sum|^2 - n) / (n (n - 1))
    m = len(unit)
    mean_cosine = float((total @ total - m) / (m * (m - 1)))
    singular = torch.linalg.svdvals(x - x.mean(dim=0))
    p = singular / singular.sum().clamp_min(1e-12)
    p = p[p > 0]
    effective_rank = float(torch.exp(-(p * p.log()).sum())) if len(p) else 0.0
    return EmbeddingHealth(
        rows=m,
        dim=dim,
        mean_cosine=mean_cosine,
        effective_rank=effective_rank,
        effective_rank_ratio=min(1.0, effective_rank / dim),
    )
