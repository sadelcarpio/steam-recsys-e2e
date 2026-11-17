"""Ranker matrices, LightGBM `lambdarank` and ranking metrics (spec 9).

`build_groups` turns query groups + the retriever's candidates into one flat feature matrix
(rows of a group are contiguous, in retrieval order). Candidates the user reviews positively
later, and slots without a candidate (-inf score), are left out. `RankingMetrics` count a group
whose game is not among its candidates as a miss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from steam_training.contracts import RankingMetrics
from steam_training.ranker.data import RankerRows
from steam_training.ranker.features import (
    FEATURES,
    GameTable,
    PopularityTable,
    compute_features,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Groups:
    """Flat ranking data: group i is rows offsets[i]:offsets[i + 1]."""

    features: np.ndarray  # float32 [rows, len(FEATURES)]
    label: np.ndarray  # int8 [rows]
    offsets: np.ndarray  # int64 [groups + 1]
    history_size: np.ndarray  # int64 [groups]: liked games the group's features read

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def has_target(self) -> np.ndarray:
        """bool [groups]: the group's game is among its candidates."""
        owner = np.repeat(np.arange(len(self)), self.sizes)
        return np.bincount(owner, weights=self.label, minlength=len(self)) > 0

    def take(self, groups: np.ndarray) -> Groups:
        sizes = self.sizes[groups]
        offsets = np.zeros(len(groups) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        within = np.arange(offsets[-1]) - np.repeat(offsets[:-1], sizes)
        rows = np.repeat(self.offsets[groups], sizes) + within
        return Groups(self.features[rows], self.label[rows], offsets, self.history_size[groups])


def build_groups(
    rows: RankerRows,
    candidates: np.ndarray,
    scores: np.ndarray,
    games: GameTable,
    popularity: PopularityTable,
    history_length: int,
    chunk: int = 256,
) -> Groups:
    """Features + labels of every group (`candidates` / `scores` [groups, k], best first).
    The user-side features read the first `history_length` games of each history."""
    features, labels, sizes, history_size = [], [], [], []
    num_games = games.num_games
    for start in range(0, len(rows), chunk):
        part = rows.take(np.arange(start, min(start + chunk, len(rows))))
        cand, score = candidates[start : start + len(part)], scores[start : start + len(part)]
        history = part.history[:, :history_length]
        x = compute_features(history, cand, score, part.seconds, games, popularity)
        later_owner = np.repeat(np.arange(len(part)), np.diff(part.later.offsets))
        later_key = later_owner * num_games + part.later.values
        cand_key = np.arange(len(part))[:, None] * num_games + cand
        keep = np.isfinite(score) & ~np.isin(cand_key, later_key)
        features.append(x[keep])
        labels.append((cand == part.target[:, None])[keep].astype(np.int8))
        sizes.append(keep.sum(axis=1))
        history_size.append((history >= 2).sum(axis=1))
    sizes_all = np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)
    offsets = np.zeros(len(sizes_all) + 1, dtype=np.int64)
    np.cumsum(sizes_all, out=offsets[1:])
    return Groups(
        np.concatenate(features) if features else np.zeros((0, len(FEATURES)), np.float32),
        np.concatenate(labels) if labels else np.zeros(0, np.int8),
        offsets,
        np.concatenate(history_size) if history_size else np.zeros(0, np.int64),
    )


def target_positions(score: np.ndarray, groups: Groups) -> np.ndarray:
    """0-based position of each group's game when its rows are sorted by `score` (descending;
    ties keep retrieval order); -1 when the game is not among the candidates."""
    n = len(groups)
    owner = np.repeat(np.arange(n), groups.sizes)
    has = groups.has_target
    pos_row = np.full(n, -1, dtype=np.int64)
    positive = np.flatnonzero(groups.label == 1)
    pos_row[owner[positive]] = positive
    target_score = np.where(has, score[np.maximum(pos_row, 0)], np.inf)[owner]
    target_row = pos_row[owner]
    ahead = (score > target_score) | (
        (score == target_score) & (np.arange(len(score)) < target_row)
    )
    position = np.bincount(owner, weights=ahead, minlength=n).astype(np.int64)
    return np.where(has, position, -1)


def ranking_metrics(position: np.ndarray, ks: list[int]) -> RankingMetrics:
    """Recall@k and NDCG@k (one relevant game per group) from `target_positions`."""
    n = len(position)
    found = position >= 0
    recall, ndcg, mrr = {}, {}, {}
    gain = np.where(found, 1 / np.log2(np.maximum(position, 0) + 2), 0)
    reciprocal = np.where(found, 1 / (np.maximum(position, 0) + 1), 0)
    for k in sorted(ks):
        inside = found & (position < k)
        recall[k] = float(inside.mean()) if n else 0.0
        ndcg[k] = float(np.where(inside, gain, 0).mean()) if n else 0.0
        mrr[k] = float(np.where(inside, reciprocal, 0).mean()) if n else 0.0
    return RankingMetrics(groups=n, recall=recall, ndcg=ndcg, mrr=mrr)


def retrieval_score(groups: Groups) -> np.ndarray:
    """The retrieval order as a score (higher = better): minus the rank feature."""
    return -groups.features[:, FEATURES.index("rank")]


@dataclass(frozen=True)
class FitResult:
    booster: object  # lightgbm.Booster
    features: list[str]
    best_iteration: int

    def predict(self, groups: Groups) -> np.ndarray:
        return self.predict_features(groups.features)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Scores of rows of a full `FEATURES` matrix ([rows, len(FEATURES)])."""
        idx = [FEATURES.index(f) for f in self.features]
        return self.booster.predict(features[:, idx], num_iteration=self.best_iteration)


def fit_lambdarank(
    train: Groups,
    features: list[str],
    *,
    early_stopping_fraction: float,
    top_k: int,
    num_leaves: int,
    learning_rate: float,
    max_rounds: int,
    early_stopping_rounds: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    num_threads: int,
    seed: int,
) -> FitResult:
    """LightGBM lambdarank on `train` (groups in time order); the last `early_stopping_fraction`
    of the groups stops it early on NDCG@top_k."""
    import lightgbm as lgb

    idx = [FEATURES.index(f) for f in features]
    cut = max(1, min(len(train) - 1, int(round(len(train) * (1 - early_stopping_fraction)))))
    fit, stop = train.take(np.arange(cut)), train.take(np.arange(cut, len(train)))
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [top_k],
        "lambdarank_truncation_level": top_k + 10,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_threads": num_threads,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(fit.features[:, idx], fit.label, group=fit.sizes, feature_name=features)
    dstop = lgb.Dataset(
        stop.features[:, idx], stop.label, group=stop.sizes, feature_name=features, reference=dtrain
    )
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dstop],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best = booster.best_iteration or booster.current_iteration()
    log.info("lambdarank: %d features, best iteration %d", len(features), best)
    return FitResult(booster, features, int(best))
