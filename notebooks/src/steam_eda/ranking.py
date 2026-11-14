"""Ranking EDA (spec 9): which feature families separate the true next game from a retriever's
top K? Building blocks of `02_ranking_features`.

The champion's towers don't run on the public dataset, so a **proxy retriever** produces the
candidates: SVD item embeddings fit on positives before `t1` (games reweighted by
1/sqrt(popularity)), the user as the mean of the last `history` positives, plus `beta` times the
standardized log of the game's positives in the 13 weeks before the month of t (a recency term,
so games released after `t1` can be retrieved).

```python
arrays = GameArrays.build(games)                      # tags, genres, developers, price, ...
pop = WeeklyPopularity.build(weekly_counts(scan("reviews")), arrays.gid2gi)
seq = Sequences.build(reviews, arrays.gid2gi)         # one user sample, sorted by user, time
groups = build_groups(seq, t1, t2, cap=20)            # one query group per positive after t1
cands, scores = retrieve(seq, groups, retriever, k=100)
X, y, qid = group_features(seq, groups, cands, scores, arrays, pop, emb, ...)
```

Every feature of a group uses only the user's reviews strictly before t and game counts of the
weeks before t's week (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

SECONDS_PER_WEEK = 7 * 86_400

FAMILIES: dict[str, list[str]] = {
    "R": ["score", "rank"],
    "C": ["log_pos", "ratio", "is_free", "log_price", "age_days", "log_pos_30d", "log_pos_90d"],
    "A": ["tag_cos_full", "tag_cos_last", "dev_share", "pub_share", "genre_cos", "categ_cos"],
    "H": ["emb_max_full", "emb_mean_full", "emb_max_last", "emb_max_older", "emb_mean_older"],
    "N": ["neg_emb_max", "neg_tag_cos", "dev_neg_share"],
    "F": ["pop_fit", "year_fit", "price_fit", "free_share"],
    "U": ["n_pos", "n_neg", "pos_rate", "days_since_last", "hist_gt_last"],
    "L": ["lang_share"],
    "P": ["tag_cos_playtime", "emb_max_playtime"],
}
FAMILY_NAMES = {
    "R": "retrieval",
    "C": "candidate",
    "A": "affinity",
    "H": "history",
    "N": "negatives",
    "F": "fit",
    "U": "user",
    "L": "language (needs ETL)",
    "P": "playtime (needs ETL)",
}
FEATURES = [f for fs in FAMILIES.values() for f in fs]


def _week(ts: np.ndarray | int) -> np.ndarray | int:
    return ts // SECONDS_PER_WEEK


# ------------------------------------------------------------------ games


@dataclass
class GameArrays:
    """Per-game arrays indexed by `gi` (row of `games`)."""

    gid2gi: dict[int, int]
    tags: np.ndarray  # float32 [n, tags], tag weights, L2-normalized rows
    genres: np.ndarray  # float32 multi-hot
    categories: np.ndarray
    developer: np.ndarray  # int64 id of the first developer (-1 = none)
    publisher: np.ndarray
    is_free: np.ndarray  # bool
    log_price: np.ndarray  # float32
    release_day: np.ndarray  # float64 days since epoch (NaN = unknown)

    @property
    def n(self) -> int:
        return len(self.is_free)

    @classmethod
    def build(cls, games: pl.DataFrame, top_tags: int = 20) -> GameArrays:
        """`games` from `load_games()` (tags + parsed `release_date`)."""
        n = games.height
        gid2gi = {g: i for i, g in enumerate(games["game_id"].to_list())}
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        names = games["tag_names"].to_list()
        weights = games["tag_weights"].to_list()
        for gi, (ns, ws) in enumerate(zip(names, weights, strict=True)):
            for name, w in list(zip(ns or [], ws or [], strict=True))[:top_tags]:
                rows.append(gi)
                cols.append(vocab.setdefault(name, len(vocab)))
                vals.append(float(w))
        tags = np.zeros((n, max(len(vocab), 1)), dtype=np.float32)
        tags[rows, cols] = vals
        tags /= np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1e-9)
        return cls(
            gid2gi=gid2gi,
            tags=tags,
            genres=_multihot(games["game_genres"].to_list()),
            categories=_multihot(games["game_categories"].to_list()),
            developer=_first_id(games["game_developers"].to_list()),
            publisher=_first_id(games["game_publishers"].to_list()),
            is_free=games["game_is_free"].fill_null(False).to_numpy(),
            log_price=np.log1p(games["game_price"].fill_null(0).to_numpy()).astype(np.float32),
            release_day=games["release_date"]
            .cast(pl.Datetime("us"))
            .dt.epoch("d")
            .cast(pl.Float64)
            .fill_null(np.nan)
            .to_numpy(),
        )


def _multihot(lists: list) -> np.ndarray:
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for gi, values in enumerate(lists):
        for v in values or []:
            rows.append(gi)
            cols.append(vocab.setdefault(v, len(vocab)))
    m = np.zeros((len(lists), max(len(vocab), 1)), dtype=np.float32)
    m[rows, cols] = 1
    return m


def _first_id(lists: list) -> np.ndarray:
    vocab: dict[str, int] = {}
    out = np.full(len(lists), -1, dtype=np.int64)
    for gi, values in enumerate(lists):
        if values:
            out[gi] = vocab.setdefault(values[0], len(vocab))
    return out


def with_gi(df: pl.DataFrame, gid2gi: dict[int, int]) -> pl.DataFrame:
    """Adds `gi` (games row) and drops reviews of games missing from `games`."""
    return df.with_columns(
        gi=pl.col("game_id").replace_strict(gid2gi, default=None, return_dtype=pl.Int64)
    ).drop_nulls("gi")


# ------------------------------------------------------------------ popularity as of t


def weekly_counts(reviews: pl.LazyFrame) -> pl.DataFrame:
    """Reviews and positives per (game_id, week) over every review."""
    return (
        reviews.select(
            "game_id",
            "is_positive",
            week=(pl.col("reviewed_at").cast(pl.Datetime("us")).dt.epoch("s") // SECONDS_PER_WEEK),
        )
        .group_by("game_id", "week")
        .agg(pos=pl.col("is_positive").sum().cast(pl.Int64), n=pl.len().cast(pl.Int64))
        .collect(engine="streaming")
    )


@dataclass
class WeeklyPopularity:
    """Cumulative positives / reviews per game, looked up as of the weeks before a given week."""

    gi: np.ndarray
    week: np.ndarray
    pos: np.ndarray
    cum_pos: np.ndarray
    cum_n: np.ndarray
    start: np.ndarray
    end: np.ndarray

    @classmethod
    def build(cls, weekly: pl.DataFrame, gid2gi: dict[int, int]) -> WeeklyPopularity:
        w = with_gi(weekly, gid2gi).sort("gi", "week")
        gi = w["gi"].to_numpy()
        pos = w["pos"].to_numpy().astype(np.float64)
        n = w["n"].to_numpy().astype(np.float64)
        first = np.r_[True, gi[1:] != gi[:-1]]
        seg = np.cumsum(first) - 1
        offsets = np.r_[0, np.cumsum(pos)][np.flatnonzero(first)]
        offsets_n = np.r_[0, np.cumsum(n)][np.flatnonzero(first)]
        n_games = len(gid2gi)
        return cls(
            gi=gi,
            week=w["week"].to_numpy(),
            pos=pos,
            cum_pos=np.cumsum(pos) - offsets[seg],
            cum_n=np.cumsum(n) - offsets_n[seg],
            start=np.searchsorted(gi, np.arange(n_games)),
            end=np.searchsorted(gi, np.arange(n_games), side="right"),
        )

    def _asof(self, gis: np.ndarray, week: int, values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(gis))
        for j, g in enumerate(gis):
            s, e = self.start[g], self.end[g]
            i = np.searchsorted(self.week[s:e], week) - 1  # weeks strictly before `week`
            if i >= 0:
                out[j] = values[s + i]
        return out

    def positives(self, gis: np.ndarray, week: int) -> np.ndarray:
        return self._asof(gis, week, self.cum_pos)

    def reviews(self, gis: np.ndarray, week: int) -> np.ndarray:
        return self._asof(gis, week, self.cum_n)

    def window(self, week: int, weeks: int, n_games: int) -> np.ndarray:
        """Positives of every game in the `weeks` weeks before `week`."""
        sel = (self.week >= week - weeks) & (self.week < week)
        return np.bincount(self.gi[sel], weights=self.pos[sel], minlength=n_games)


# ------------------------------------------------------------------ user sequences and groups


@dataclass
class Sequences:
    """A user sample's reviews sorted by user then time; `bounds[u]:bounds[u+1]` is user u."""

    user: np.ndarray
    gi: np.ndarray
    positive: np.ndarray
    ts: np.ndarray  # seconds since epoch
    language: np.ndarray
    log_hours: np.ndarray  # log1p(playtime at review in hours)
    bounds: np.ndarray

    @classmethod
    def build(cls, reviews: pl.DataFrame, gid2gi: dict[int, int]) -> Sequences:
        r = with_gi(reviews, gid2gi).sort("user_id", "reviewed_at")
        user = r["user_id"].to_numpy()
        return cls(
            user=user,
            gi=r["gi"].to_numpy(),
            positive=r["is_positive"].to_numpy(),
            ts=r["reviewed_at"].cast(pl.Datetime("us")).dt.epoch("s").to_numpy(),
            language=r["language"].fill_null("").to_numpy(),
            log_hours=np.log1p(r["playtime_at_review"].fill_null(0).to_numpy() / 60).astype(
                np.float32
            ),
            bounds=np.flatnonzero(np.r_[True, user[1:] != user[:-1], True]),
        )


@dataclass
class Groups:
    """Query groups: rows (a, b, i) = user slice [a, b) and the target review i."""

    rows: np.ndarray  # int64 [n, 3]

    def __len__(self) -> int:
        return len(self.rows)

    def take(self, idx: np.ndarray) -> Groups:
        return Groups(self.rows[idx])


def build_groups(
    seq: Sequences,
    t1: datetime,
    t2: datetime,
    cap: int,
    seed: int = 0,
) -> tuple[Groups, Groups]:
    """One group per positive review at t >= t1 by a user with a positive review before t
    (warm rows: inference never scores an empty history), at most `cap` per user. Train: t < t2,
    test: t >= t2. Train is sorted by time (the last part is the early-stopping slice)."""
    rng = np.random.default_rng(seed)
    t1s, t2s = t1.timestamp(), t2.timestamp()
    out = []
    for a, b in zip(seq.bounds[:-1], seq.bounds[1:], strict=True):
        ts, pos = seq.ts[a:b], seq.positive[a:b]
        targets = np.flatnonzero(pos & (ts >= t1s))
        if not len(targets):
            continue
        # first positive time: a target is warm if an earlier positive exists strictly before it
        first_pos = ts[np.flatnonzero(pos)[0]]
        targets = targets[ts[targets] > first_pos]
        if len(targets) > cap:
            targets = rng.choice(targets, cap, replace=False)
        out += [(a, b, a + t) for t in targets]
    rows = np.array(out, dtype=np.int64).reshape(-1, 3)
    test = seq.ts[rows[:, 2]] >= t2s
    train = rows[~test]
    train = train[np.argsort(seq.ts[train[:, 2]], kind="stable")]
    return Groups(train), Groups(rows[test])


def subsample(groups: Groups, n: int, seed: int = 0) -> Groups:
    """At most n groups, keeping their order."""
    if len(groups) <= n:
        return groups
    rng = np.random.default_rng(seed)
    return groups.take(np.sort(rng.choice(len(groups), n, replace=False)))


# ------------------------------------------------------------------ proxy retriever


def svd_embeddings(positives: pl.DataFrame, n_games: int, dim: int = 64, seed: int = 0):
    """Item embeddings (L2-normalized, zero for games with < 5 positives) from a truncated SVD
    of the user × game matrix of positives (`user_id`, `gi`), games weighted 1/sqrt(count).
    Returns (embeddings, positives per game)."""
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd

    user = positives["user_id"].rank("dense").cast(pl.Int64).to_numpy() - 1
    gi = positives["gi"].to_numpy()
    counts = np.bincount(gi, minlength=n_games)
    w = 1.0 / np.sqrt(np.maximum(counts[gi], 1))
    m = sp.csr_matrix((w, (user, gi)), shape=(user.max() + 1, n_games), dtype=np.float32)
    dim = max(1, min(dim, min(m.shape) - 1))
    _, s, vt = randomized_svd(m, n_components=dim, n_iter=4, random_state=seed)
    emb = (vt.T * np.sqrt(s)).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    emb[counts < 5] = 0
    return emb, counts


@dataclass
class ProxyRetriever:
    emb: np.ndarray  # [n_games, d]
    pool: np.ndarray  # candidate games (gi)
    month_week: np.ndarray  # first week of each month
    recent: np.ndarray  # [months, pool] standardized log recent positives
    history: int
    beta: float = 0.0

    @classmethod
    def build(
        cls,
        emb: np.ndarray,
        counts_before: np.ndarray,
        pop: WeeklyPopularity,
        months: list[datetime],
        history: int,
        min_pool_positives: float,
        recent_weeks: int = 13,
    ) -> ProxyRetriever:
        n_games = len(emb)
        month_week = np.array([_week(int(m.timestamp())) for m in months])
        recent = np.stack([pop.window(w, recent_weeks, n_games) for w in month_week])
        pool = np.flatnonzero((counts_before >= min_pool_positives) | (recent.max(0) >= 20))
        lp = np.log1p(recent[:, pool])
        lp = (lp - lp.mean()) / max(lp.std(), 1e-9)
        return cls(emb, pool, month_week, lp.astype(np.float32), history)

    def retrieve(self, seq: Sequences, groups: Groups, k: int, batch: int = 2048):
        """Top k pool games per group (games u reviewed before t excluded), best first."""
        k = min(k, len(self.pool) - 1)
        slot = np.full(len(self.emb), -1)
        slot[self.pool] = np.arange(len(self.pool))
        pool_emb = self.emb[self.pool]
        cands, scores = [], []
        for s in range(0, len(groups), batch):
            rows = groups.rows[s : s + batch]
            user = np.zeros((len(rows), self.emb.shape[1]), dtype=np.float32)
            for j, (a, _, i) in enumerate(rows):
                before = seq.ts[a:i] < seq.ts[i]
                hist = seq.gi[a:i][before & seq.positive[a:i]][-self.history :]
                user[j] = self.emb[hist].mean(0)
            user /= np.maximum(np.linalg.norm(user, axis=1, keepdims=True), 1e-9)
            month = np.searchsorted(self.month_week, _week(seq.ts[rows[:, 2]]), side="right") - 1
            sc = user @ pool_emb.T + self.beta * self.recent[np.maximum(month, 0)]
            for j, (a, _, i) in enumerate(rows):
                seen = slot[seq.gi[a:i][seq.ts[a:i] < seq.ts[i]]]
                sc[j, seen[seen >= 0]] = -np.inf
            top = np.argpartition(-sc, k, axis=1)[:, :k]
            top_scores = np.take_along_axis(sc, top, 1)
            order = np.argsort(-top_scores, axis=1)
            cands.append(self.pool[np.take_along_axis(top, order, 1)])
            scores.append(np.take_along_axis(top_scores, order, 1))
        if not cands:
            return np.zeros((0, k), dtype=np.int64), np.zeros((0, k), dtype=np.float32)
        return np.vstack(cands), np.vstack(scores)


def hits(seq: Sequences, groups: Groups, cands: np.ndarray) -> np.ndarray:
    """bool [groups, k]: candidate j is the group's target."""
    return cands == seq.gi[groups.rows[:, 2]][:, None]


# ------------------------------------------------------------------ features


def group_features(
    seq: Sequences,
    groups: Groups,
    cands: np.ndarray,
    scores: np.ndarray,
    games: GameArrays,
    pop: WeeklyPopularity,
    emb: np.ndarray,
    language_share: dict[tuple[int, str], float],
    history: int,
    keep_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix (columns `FEATURES`), labels and query ids, one block per group.

    Candidates u reviews positively later (other than the target) are dropped (not negatives).
    Groups without the target among the candidates are dropped unless `keep_missing` (test: they
    count as misses)."""
    blocks, labels, qids = [], [], []
    for q, ((a, b, i), cand, sc) in enumerate(zip(groups.rows, cands, scores, strict=True)):
        t = seq.ts[i]
        before = seq.ts[a:b] < t
        hist, hpos = seq.gi[a:b][before], seq.positive[a:b][before]
        later = set(seq.gi[a:b][~before & seq.positive[a:b]].tolist()) - {int(seq.gi[i])}
        keep = np.array([c not in later for c in cand], dtype=bool)
        cand, sc = cand[keep], sc[keep]
        label = (cand == seq.gi[i]).astype(np.int8)
        if not label.any() and not keep_missing:
            continue
        f = _features(
            cand, sc, t, seq.ts[a:b][before].max(), hist, hpos, seq.log_hours[a:b][before],
            seq.language[a:b][before], games, pop, emb, language_share, history,
        )  # fmt: skip
        blocks.append(np.column_stack([np.asarray(f[k], dtype=np.float32) for k in FEATURES]))
        labels.append(label)
        qids.append(np.full(len(cand), q))
    if not blocks:
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros(0, np.int8), np.zeros(0, int)
    return np.vstack(blocks), np.concatenate(labels), np.concatenate(qids)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _share(ids: np.ndarray, of: np.ndarray) -> np.ndarray:
    """For each id in `of`: share of `ids` equal to it (0 for unknown ids or empty `ids`)."""
    if not len(ids):
        return np.zeros(len(of))
    values, counts = np.unique(ids, return_counts=True)
    lookup = dict(zip(values.tolist(), (counts / len(ids)).tolist(), strict=True))
    return np.array([lookup.get(x, 0.0) if x >= 0 else 0.0 for x in of.tolist()])


def _features(
    cand, sc, t, last_ts, hist, hpos, hours, langs, games, pop, emb, language_share, history
):
    n = len(cand)
    nan = np.full(n, np.nan)
    week = int(_week(t))
    pos_g, neg_g = hist[hpos], hist[~hpos]
    cum_pos = pop.positives(cand, week)
    cum_n = pop.reviews(cand, week)
    f: dict[str, np.ndarray] = {}
    # retrieval
    f["score"], f["rank"] = sc, np.arange(n, dtype=np.float32)
    # candidate, as of t
    f["log_pos"] = np.log1p(cum_pos)
    f["ratio"] = (cum_pos + 1) / (cum_n + 2)
    f["is_free"] = games.is_free[cand].astype(np.float32)
    f["log_price"] = games.log_price[cand]
    f["age_days"] = t / 86_400 - games.release_day[cand]
    f["log_pos_30d"] = np.log1p(cum_pos - pop.positives(cand, week - 4))
    f["log_pos_90d"] = np.log1p(cum_pos - pop.positives(cand, week - 13))
    # affinity
    tags = games.tags[cand]
    f["tag_cos_full"] = tags @ _unit(games.tags[pos_g].sum(0))
    f["tag_cos_last"] = tags @ _unit(games.tags[pos_g[-history:]].sum(0))
    f["dev_share"] = _share(games.developer[pos_g], games.developer[cand])
    f["pub_share"] = _share(games.publisher[pos_g], games.publisher[cand])
    g = games.genres[cand]
    f["genre_cos"] = g @ games.genres[pos_g].mean(0) / np.maximum(g.sum(1), 1)
    c = games.categories[cand]
    f["categ_cos"] = c @ games.categories[pos_g].mean(0) / np.maximum(c.sum(1), 1)
    # history (embedding similarity), full / last `history` / older
    sim = emb[cand] @ emb[pos_g].T
    f["emb_max_full"], f["emb_mean_full"] = sim.max(1), sim.mean(1)
    f["emb_max_last"] = sim[:, -history:].max(1)
    older = sim[:, :-history] if len(pos_g) > history else None
    f["emb_max_older"] = older.max(1) if older is not None else nan
    f["emb_mean_older"] = older.mean(1) if older is not None else nan
    # negatives
    if len(neg_g):
        f["neg_emb_max"] = (emb[cand] @ emb[neg_g].T).max(1)
        f["neg_tag_cos"] = tags @ _unit(games.tags[neg_g].sum(0))
        f["dev_neg_share"] = _share(games.developer[neg_g], games.developer[cand])
    else:
        f["neg_emb_max"] = f["neg_tag_cos"] = f["dev_neg_share"] = nan
    # fit
    f["pop_fit"] = f["log_pos"] - np.log1p(pop.positives(pos_g, week)).mean()
    rel = games.release_day[pos_g]
    f["year_fit"] = (
        (games.release_day[cand] - np.nanmedian(rel)) / 365 if np.isfinite(rel).any() else nan
    )
    f["price_fit"] = games.log_price[cand] - np.median(games.log_price[pos_g])
    f["free_share"] = np.full(n, games.is_free[pos_g].mean())
    # user (constant in the group: only useful through splits)
    f["n_pos"], f["n_neg"] = np.full(n, len(pos_g)), np.full(n, len(neg_g))
    f["pos_rate"] = np.full(n, len(pos_g) / len(hist))
    f["days_since_last"] = np.full(n, (t - last_ts) / 86_400)
    f["hist_gt_last"] = np.full(n, float(len(pos_g) > history))
    # language: share of the candidate's reviews (before t1) in u's dominant language
    values, counts = np.unique(langs, return_counts=True)
    dominant = values[counts.argmax()]
    f["lang_share"] = np.array([language_share.get((int(x), dominant), 0.0) for x in cand])
    # playtime-weighted profile
    w = hours[hpos] + 1e-3
    f["tag_cos_playtime"] = tags @ _unit((games.tags[pos_g] * w[:, None]).sum(0))
    f["emb_max_playtime"] = (sim * (w / w.max())[None, :]).max(1)
    return f


# ------------------------------------------------------------------ evaluation


def group_sizes(qid: np.ndarray) -> np.ndarray:
    """Rows per group, in order of appearance (qid is grouped)."""
    if not len(qid):
        return np.zeros(0, dtype=int)
    edges = np.flatnonzero(np.r_[True, qid[1:] != qid[:-1], True])
    return np.diff(edges)


def _starts(qid: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(group_sizes(qid))]


def ranking_metrics(
    score: np.ndarray, label: np.ndarray, qid: np.ndarray, ks: tuple[int, ...] = (10, 20)
) -> dict[str, float]:
    """recall@k and NDCG@max(k) (one relevant item per group), over every group (groups without
    the positive count as misses)."""
    st = _starts(qid)
    n = len(st) - 1
    out = {f"recall@{k}": 0.0 for k in ks}
    kmax = max(ks)
    out[f"ndcg@{kmax}"] = 0.0
    for a, b in zip(st[:-1], st[1:], strict=True):
        lab = label[a:b]
        if not lab.any():
            continue
        r = int(np.flatnonzero(lab[np.argsort(-score[a:b], kind="stable")])[0])
        for k in ks:
            out[f"recall@{k}"] += r < k
        if r < kmax:
            out[f"ndcg@{kmax}"] += 1 / np.log2(r + 2)
    return {k: (v / n if n else float("nan")) for k, v in out.items()}


def group_auc(feature: np.ndarray, label: np.ndarray, qid: np.ndarray) -> tuple[float, int]:
    """Mean over groups with a positive of P(feature[positive] > feature[negative]) (ties ½,
    NaN skipped), and the number of groups used. 0.5 = no signal; < 0.5 = higher means less
    likely. Constant-in-group features are 0.5 by construction."""
    st = _starts(qid)
    values = []
    for a, b in zip(st[:-1], st[1:], strict=True):
        lab, x = label[a:b], feature[a:b]
        if not lab.any():
            continue
        p = x[lab == 1][0]
        neg = x[lab == 0]
        neg = neg[~np.isnan(neg)]
        if np.isnan(p) or not len(neg):
            continue
        values.append(((neg < p).sum() + 0.5 * (neg == p).sum()) / len(neg))
    return (float(np.mean(values)) if values else float("nan")), len(values)
