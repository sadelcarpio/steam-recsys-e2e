"""Pure helpers to split appids across workers."""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence


def chunk_by_size(items: Sequence[int], max_size: int) -> list[list[int]]:
    """Split into the fewest chunks of at most `max_size`, balanced in size."""
    if not items:
        return []
    n = math.ceil(len(items) / max_size)
    base, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + base + (1 if i < extra else 0)
        chunks.append(list(items[start:end]))
        start = end
    return chunks


def balance_by_weight(
    items: Sequence[int], weights: Mapping[int, float], n_bins: int
) -> list[list[int]]:
    """Greedy longest-processing-time assignment; empty bins are dropped, bins are sorted."""
    heap: list[tuple[float, int]] = [(0.0, i) for i in range(n_bins)]
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for item in sorted(items, key=lambda a: (-weights.get(a, 1.0), a)):
        load, idx = heapq.heappop(heap)
        bins[idx].append(item)
        heapq.heappush(heap, (load + weights.get(item, 1.0), idx))
    return [sorted(b) for b in bins if b]


def estimated_review_requests(total_reviews: int | None, max_reviews_per_game: int) -> float:
    """Rough request count to scrape a game's reviews: 1 + pages of (capped) reviews."""
    total = total_reviews or 0
    if max_reviews_per_game:
        total = min(total, max_reviews_per_game)
    return 1.0 + total / 100
