"""Text models for the embedding and sentiment questions, with light fallbacks.

The real models need the optional group (`uv sync --group nlp`: CPU torch,
sentence-transformers, transformers) and a first download from the Hugging Face hub. Without
it, `embed_texts` falls back to TF-IDF + truncated SVD (a lexical baseline, worth comparing
anyway) and `sentiment` returns None, so the notebook still runs end to end.
"""

from __future__ import annotations

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def has_nlp() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def tfidf_embed(texts: list[str], dim: int = 128, seed: int = 0) -> np.ndarray:
    """Lexical baseline: TF-IDF (1-2 grams) → truncated SVD → L2-normalized rows."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
        min_df=2 if len(texts) > 50 else 1,
    )  # fmt: skip
    matrix = tfidf.fit_transform(texts)
    dim = max(1, min(dim, matrix.shape[1] - 1, len(texts) - 1))
    reduced = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(matrix)
    return normalize(reduced).astype(np.float32)


def embed_texts(
    texts: list[str], method: str = "auto", batch_size: int = 128
) -> tuple[np.ndarray, str]:
    """L2-normalized embeddings of `texts` and the method used (`"minilm"` or
    `"tfidf-svd"`). `method`: "auto" (MiniLM when installed), "minilm" or "tfidf"."""
    if method == "tfidf" or (method == "auto" and not has_nlp()):
        return tfidf_embed(texts), "tfidf-svd"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    emb = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )
    return np.asarray(emb, dtype=np.float32), "minilm"


def sentiment(texts: list[str], batch_size: int = 64) -> np.ndarray | None:
    """P(positive) of each (English) text with an SST-2 classifier, truncated to 256 tokens;
    None without the nlp group."""
    if not has_nlp():
        return None
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis", model=SENTIMENT_MODEL, device=-1, truncation=True, max_length=256
    )
    out = clf(texts, batch_size=batch_size)
    return np.array(
        [o["score"] if o["label"] == "POSITIVE" else 1 - o["score"] for o in out], dtype=float
    )


def knn(emb: np.ndarray, k: int = 10, block: int = 4096) -> np.ndarray:
    """Indices of the `k` nearest rows by cosine (rows L2-normalized), self excluded."""
    k = min(k, len(emb) - 1)
    out = np.empty((len(emb), k), dtype=np.int64)
    for start in range(0, len(emb), block):
        sims = emb[start : start + block] @ emb.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out[start : start + block] = np.take_along_axis(top, order, axis=1)
    return out
