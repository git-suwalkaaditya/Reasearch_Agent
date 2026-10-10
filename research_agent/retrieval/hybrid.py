"""Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with Reciprocal Rank Fusion.

Why hybrid? BM25 nails exact terms (method names, dataset names, acronyms); dense vectors
catch paraphrases ("privacy budget" vs "epsilon"). RRF merges the two rankings without
having to calibrate their very different score scales:  score(d) = sum_i 1 / (k + rank_i(d)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..memory import PersistentStore
from ..schemas import Chunk
from ..utils import tokenize
from .bm25 import BM25
from .embeddings import Embedder


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class HybridIndex:
    def __init__(self, embedder: Embedder, store: Optional[PersistentStore] = None, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._titles: dict[str, str] = {}
        self._dirty = True
        self._bm25 = BM25()
        self._matrix = np.zeros((0, 1), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], title: str = "") -> None:
        for c in chunks:
            if c.chunk_id not in self._ids:
                self.chunks.append(c)
                self._ids.add(c.chunk_id)
                if title:
                    self._titles[c.paper_id] = title
        self._dirty = True

    def _doc_text(self, c: Chunk) -> str:
        # Prefixing title + section gives every chunk some document-level context.
        return f"{self._titles.get(c.paper_id, '')} | {c.section} | {c.text}"

    def _build(self) -> None:
        texts = [self._doc_text(c) for c in self.chunks]
        self._bm25.fit([tokenize(t) for t in texts])
        if self.embedder.cacheable and self.store is not None:
            ids = [c.chunk_id for c in self.chunks]
            cached = self.store.get_embeddings(ids, self.embedder.cache_key)
            missing = [i for i, cid in enumerate(ids) if cid not in cached]
            if missing:
                vecs = self.embedder.encode([texts[i] for i in missing])
                new = {ids[i]: vecs[j] for j, i in enumerate(missing)}
                self.store.put_embeddings(new, self.embedder.cache_key)
                cached.update(new)
            self._matrix = np.vstack([cached[cid] for cid in ids]).astype(np.float32)
        else:
            self.embedder.fit(texts)
            self._matrix = self.embedder.encode(texts)
        self._dirty = False

    def search(self, query: str, k: int = 20, paper_ids: Optional[set[str]] = None) -> list[Hit]:
        if not self.chunks:
            return []
        if self._dirty:
            self._build()
        mask = np.array([paper_ids is None or c.paper_id in paper_ids for c in self.chunks])
        if not mask.any():
            return []
        bm = self._bm25.scores(tokenize(query))
        dense = self._matrix @ self.embedder.encode([query])[0]
        bm[~mask], dense[~mask] = -np.inf, -np.inf

        n_valid = int(mask.sum())
        bm_order = [i for i in np.argsort(-bm)[:n_valid] if bm[i] > 0]  # BM25 0 = no term overlap
        dn_order = list(np.argsort(-dense)[:n_valid])
        bm_rank = {i: r for r, i in enumerate(bm_order)}
        dn_rank = {i: r for r, i in enumerate(dn_order)}

        fused: dict[int, float] = {}
        for ranks in (bm_rank, dn_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], bm_rank.get(i), dn_rank.get(i)) for i in top]
