"""Spec 9, phase 4: the serving pair's ranker in batch inference + the delayed-label check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pytest
from conftest import START, FakeSource, make_marts
from steam_training.contracts import (
    PairRecord,
    RankerComparison,
    RankerMetadata,
    RankerWindow,
    RankingMetrics,
)
from steam_training.ranker.features import FEATURE_SCHEMA_VERSION, FEATURES

from steam_inference import ranking
from steam_inference.features import load_inference_data
from steam_inference.online_check import ListsSample, check
from steam_inference.pipeline import run_inference
from steam_inference.writer import JsonlWriter

NOW = START + timedelta(days=60)


def ranker_marts() -> dict[str, pa.Table]:
    """The test marts + what the ranker features read: lkp_games.game_id and the weekly
    popularity mart (counted from interactions, Monday weeks)."""
    marts = make_marts()
    games = marts["lkp_games"]["game_idx"].to_pylist()
    marts["lkp_games"] = pa.table(
        {"game_idx": pa.array(games, pa.int64()), "game_id": pa.array([1000 + g for g in games])}
    )
    weekly: dict[tuple[int, object], list[int]] = {}
    for row in marts["interactions"].to_pylist():
        monday = row["timestamp"].date() - timedelta(days=row["timestamp"].weekday())
        counts = weekly.setdefault((row["game_idx"], monday), [0, 0])
        counts[0] += row["is_positive"]
        counts[1] += 1
    marts["game_popularity_weekly"] = pa.Table.from_pylist(
        [
            {"game_idx": g, "week_start": w, "positives": p, "reviews": n}
            for (g, w), (p, n) in sorted(weekly.items())
        ]
    )
    return marts


def free_first_ranker() -> str:
    """A LightGBM model on the ranker's features that scores free games first."""
    rng = np.random.default_rng(0)
    x = rng.random((400, len(FEATURES)))
    free = FEATURES.index("is_free")
    x[:, free] = rng.integers(0, 2, 400)
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(x, x[:, free], feature_name=FEATURES),
        num_boost_round=5,
    )
    return booster.model_to_string()


def metrics() -> RankingMetrics:
    return RankingMetrics(groups=1, recall={5: 0.5}, ndcg={5: 0.5})


def serve_ranker(store, retriever_id: str = "abc123", ranker_id: str = "k1") -> None:
    window = RankerWindow(
        retriever_cutoff=datetime(2024, 5, 1),
        t2=datetime(2024, 6, 1),
        end=datetime(2024, 7, 1),
        train_groups=10,
        early_stopping_groups=2,
        test_groups=5,
    )
    store.save_ranker(
        free_first_ranker(),
        RankerMetadata(
            ranker_id=ranker_id,
            created_at=datetime(2024, 7, 1, tzinfo=UTC),
            retriever_id=retriever_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=FEATURES,
            candidates=10,
            top_k=5,
            history_length=20,
            window=window,
            snapshots={},
            hyperparameters={},
            best_iteration=5,
            test=RankerComparison(retrieval=metrics(), ranker=metrics()),
        ),
    )
    store.copy_ranker_to_champion(ranker_id)
    store.write_pair(
        PairRecord(
            retriever_id=retriever_id,
            ranker_id=ranker_id,
            promoted_at=datetime(2024, 7, 2, tzinfo=UTC),
            reason="test",
        )
    )


def run(settings, store, tmp_path, *, mode: str, name: str, now=NOW, marts=None, **extra):
    out = tmp_path / f"{name}.jsonl"
    summary = run_inference(
        settings.model_copy(
            update={"ranker_mode": mode, "retrieval_candidates": 10, "ranker_top_k": 5, **extra}
        ),
        FakeSource(marts or ranker_marts()),
        store,
        JsonlWriter(str(out)),
        None,
        now=now,
    )
    items = [__import__("json").loads(line) for line in out.read_text().splitlines()]
    return summary, {i["user_id"]: [r["game_id"] for r in i["recommendations"]] for i in items}


def test_ranker_on_writes_the_ranker_order(settings, store, champion, tmp_path):
    serve_ranker(store)
    summary, lists = run(settings, store, tmp_path, mode="on", name="on")
    assert summary.ranker_id == "k1" and summary.ranker_mode == "on"
    assert summary.ranked_users == summary.users and summary.ranker_failures == 0
    assert summary.snapshots["game_popularity_weekly"] == 7
    for user, games in lists.items():
        if user == "__popular__" or not games:
            continue
        free = [(g - 1000) % 3 == 0 for g in games]  # game_idx % 3 == 0 is free
        assert free == sorted(free, reverse=True), (user, games)  # free games first
        assert len(games) <= 5


def test_shadow_writes_retrieval_order(settings, store, champion, tmp_path):
    _, off = run(settings, store, tmp_path, mode="off", name="off")
    serve_ranker(store)
    summary, shadow = run(settings, store, tmp_path, mode="shadow", name="shadow")
    assert shadow == off  # written lists unchanged
    assert summary.ranker_mode == "shadow" and summary.ranked_users > 0
    assert summary.ranker_overlap is not None and 0 <= summary.ranker_overlap <= 1


def test_a_ranker_of_another_retriever_is_not_used(settings, store, champion, tmp_path):
    serve_ranker(store, retriever_id="other")
    summary, _ = run(settings, store, tmp_path, mode="on", name="mismatch")
    assert summary.ranker_id is None and "other" in summary.ranker_reason


def test_without_the_popularity_mart_the_ranker_is_off(settings, store, champion, tmp_path):
    serve_ranker(store)
    marts = ranker_marts()
    del marts["game_popularity_weekly"]
    summary, _ = run(settings, store, tmp_path, mode="on", name="nomart", marts=marts)
    assert summary.ranker_id is None and "mart" in summary.ranker_reason


def test_a_failing_chunk_keeps_retrieval_order(settings, store, champion, tmp_path, monkeypatch):
    _, off = run(settings, store, tmp_path, mode="off", name="off2")
    serve_ranker(store)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ranking, "compute_features", broken)
    summary, lists = run(settings, store, tmp_path, mode="on", name="broken")
    assert summary.ranked_users == 0 and summary.ranker_failures == summary.users
    # retrieval order, cut to RANKER_TOP_K
    assert {u: g[:5] for u, g in off.items() if u != "__popular__"} == {
        u: g for u, g in lists.items() if u != "__popular__"
    }


def test_online_check_scores_the_previous_runs_lists(source):
    data = load_inference_data(source)
    user = 101  # reviews 2, 3, (4), 5, 6, 7, (8) one hour apart from START + 1 day
    sample = ListsSample(
        run_at=(START + timedelta(days=1, hours=2, minutes=30)).replace(tzinfo=UTC),
        mode="shadow",
        ranker_id="k1",
        user_id=np.array([user]),
        written=np.array([[5, 9, 6, 11]]),  # 2 of the 3 later positives (5, 6, 7)
        alternative=np.array([[9, 10, 11, 12]]),  # none
    )
    result = check(sample, data, run_at=NOW.replace(tzinfo=UTC))
    assert result is not None and result.written.lists == 1
    assert result.written.precision[10] == pytest.approx(2 / 10)
    assert result.written.recall[10] == pytest.approx(2 / 3)
    assert result.alternative.recall[10] == 0
    # AP@10: hits at positions 1 and 3 -> (1/1 + 2/3) / min(10, 3)
    assert result.written.map[10] == pytest.approx((1 + 2 / 3) / 3)


def test_runs_chain_their_online_checks(settings, store, champion, tmp_path):
    serve_ranker(store)
    first, _ = run(settings, store, tmp_path, mode="shadow", name="r1", now=START)
    assert first.online_check is None  # nothing saved before
    second, _ = run(settings, store, tmp_path, mode="shadow", name="r2", now=NOW)
    check_ = second.online_check
    assert check_ is not None and check_.previous_mode == "shadow"
    assert check_.previous_ranker_id == "k1" and check_.written.lists > 0
    keys = {o["Key"] for o in store.s3.list_objects_v2(Bucket=store.bucket)["Contents"]}
    assert "evaluation/online/lists/latest.npz" in keys
    assert any(k.startswith("evaluation/online/2024") and k.endswith(".json") for k in keys)


def _rankers(tmp_path, name: str) -> dict[str, str | None]:
    lines = (tmp_path / f"{name}.jsonl").read_text().splitlines()
    items = [__import__("json").loads(line) for line in lines]
    return {i["user_id"]: i.get("ranker_id") for i in items}


def test_stored_items_name_the_ranker_that_ordered_them(settings, store, champion, tmp_path):
    serve_ranker(store)
    run(settings, store, tmp_path, mode="on", name="named")
    rankers = _rankers(tmp_path, "named")
    assert rankers.pop("__popular__") is None
    assert rankers and set(rankers.values()) == {"k1"}
    run(settings, store, tmp_path, mode="shadow", name="unnamed")  # retrieval order written
    assert set(_rankers(tmp_path, "unnamed").values()) == {None}


def test_a_failed_chunk_is_not_named(settings, store, champion, tmp_path, monkeypatch):
    serve_ranker(store)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ranking, "compute_features", broken)
    run(settings, store, tmp_path, mode="on", name="failed")
    assert set(_rankers(tmp_path, "failed").values()) == {None}


def test_the_ranker_is_part_of_the_content_hash():
    from steam_inference.contracts import Recommendation, UserRecommendations

    item = UserRecommendations(
        user_id="1",
        recommendations=[Recommendation(game_id=1, name="a", score=0.5)],
        model_id="m",
        generated_at=NOW,
        reranked=False,
    )
    named = item.model_copy(update={"ranker_id": "k1"})
    assert named.content_hash() != item.content_hash()
    assert named.to_item()["ranker_id"] == "k1" and "ranker_id" not in item.to_item()
