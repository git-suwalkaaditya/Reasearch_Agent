"""Okapi BM25 - the classic lexical ranking function, implemented from scratch (~30 lines)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_freqs: list[Counter] = []
        self.doc_len: np.ndarray = np.zeros(0)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.doc_freqs = [Counter(d) for d in docs]
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in self.doc_freqs:
            df.update(d.keys())
        n = len(docs)
        # BM25+ style idf that never goes negative
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_freqs))
        if not self.doc_freqs or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.doc_freqs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
