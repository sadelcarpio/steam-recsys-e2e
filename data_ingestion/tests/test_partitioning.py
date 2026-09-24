from steam_ingestion.partitioning import (
    balance_by_weight,
    chunk_by_size,
    estimated_review_requests,
)


def test_chunk_by_size_uses_fewest_balanced_chunks() -> None:
    chunks = chunk_by_size(list(range(10)), 4)
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert sum(chunks, []) == list(range(10))
    assert chunk_by_size([], 4) == []
    assert chunk_by_size([1, 2], 8000) == [[1, 2]]


def test_balance_by_weight_spreads_heavy_items_and_drops_empty() -> None:
    weights = {1: 100.0, 2: 90.0, 3: 10.0, 4: 10.0}
    bins = balance_by_weight([1, 2, 3, 4], weights, 2)
    loads = sorted(sum(weights[a] for a in b) for b in bins)
    assert loads == [100.0, 110.0]
    assert sorted(sum(bins, [])) == [1, 2, 3, 4]
    assert len(balance_by_weight([1, 2], {}, 10)) == 2


def test_estimated_review_requests_caps_totals() -> None:
    assert estimated_review_requests(None, 0) == 1.0
    assert estimated_review_requests(10_000, 0) == 101.0
    assert estimated_review_requests(10_000, 2000) == 21.0
