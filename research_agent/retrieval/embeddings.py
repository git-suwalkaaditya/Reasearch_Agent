"""Dense text embeddings with three interchangeable backends.

* ``sentence-transformers`` (recommended): local neural embeddings, e.g. all-MiniLM-L6-v2.
* ``openai``: the ``/embeddings`` endpoint of any OpenAI-compatible server (OpenAI, Ollama, ...).
* ``lsa``: Latent Semantic Analysis (TF-IDF + truncated SVD) with scikit-learn. Needs no model
  download, so it is the zero-setup fallback; it is corpus-dependent, so it is refit per index.

All backends return L2-normalised float32 vectors so cosine similarity = dot product.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import requests

from ..config import Settings
from ..utils import logger, retry


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


class Embedder:
    def __init__(self, settings: Settings, backend: Optional[str] = None) -> None:
        self.s = settings
        backend = (backend or settings.embedding_backend or "auto").lower()
        if settings.offline and backend == "auto":
            backend = "lsa"
        if backend == "auto":
            backend = "sentence-transformers" if sentence_transformers_available() else "lsa"
        if backend == "sentence-transformers" and not sentence_transformers_available():
            logger.warning("sentence-transformers not installed; falling back to LSA embeddings.")
            backend = "lsa"
        self.backend = backend
        self._st_model = None
        self._vectorizer = None
        self._svd = None
        logger.info("Embedding backend: %s", self.describe())

    # -- identity -------------------------------------------------------------
    def describe(self) -> str:
        return self.backend if self.backend == "lsa" else f"{self.backend}:{self.s.embedding_model}"

    @property
    def cacheable(self) -> bool:
        """Corpus-independent embeddings can be cached in SQLite across runs."""
        return self.backend in ("sentence-transformers", "openai")

    @property
    def cache_key(self) -> str:
        return self.describe()

    # -- LSA (corpus-dependent) -------------------------------------------------------
    def fit(self, corpus: list[str]) -> None:
        if self.backend != "lsa":
            return
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=1)
        x = self._vectorizer.fit_transform(corpus)
        n_comp = min(128, x.shape[0] - 1, x.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(x) if n_comp >= 16 else None

    # -- encode -------------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "sentence-transformers":
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.s.embedding_model)
            return _normalize(self._st_model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))
        if self.backend == "openai":
            return self._encode_openai(texts)
        if self._vectorizer is None:
            self.fit(texts)
        x = self._vectorizer.transform(texts)
        dense = self._svd.transform(x) if self._svd is not None else x.toarray()
        return _normalize(dense)

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        url = self.s.llm_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"} if self.s.llm_api_key else {}
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[i : i + 64]]

            def call() -> dict:
                r = requests.post(url, headers=headers, json={"model": self.s.embedding_model, "input": batch}, timeout=120)
                r.raise_for_status()
                return r.json()

            data = retry(call, attempts=3, exceptions=(requests.RequestException,), what="embedding call")
            out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return _normalize(np.array(out))
