"""Second-stage reranking of retrieved passages.

First-stage retrieval (hybrid) is fast but coarse. A reranker looks at the query and
each passage *together*, which is far more accurate, so we run it on the top ~20 only.

* ``cross-encoder``: a small local model trained for relevance (ms-marco MiniLM).
* ``llm``: ask the LLM to score passages 0-10 (works anywhere, costs tokens).
* ``none``: keep the hybrid order.
"""
from __future__ import annotations

from ..config import Settings
from ..schemas import RerankResult
from ..utils import logger
from .embeddings import sentence_transformers_available
from .hybrid import Hit


class Reranker:
    def __init__(self, settings: Settings, llm=None, backend: str | None = None) -> None:
        backend = (backend or settings.reranker_backend or "auto").lower()
        if backend == "auto":
            backend = "cross-encoder" if (sentence_transformers_available() and not settings.offline) else ("llm" if llm else "none")
        if backend == "cross-encoder" and not sentence_transformers_available():
            backend = "llm" if llm else "none"
        self.backend = backend
        self.s = settings
        self.llm = llm
        self._model = None
        logger.info("Reranker backend: %s", backend)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits or self.backend == "none":
            return hits[:top_n]
        try:
            scores = self._cross_encoder(query, hits) if self.backend == "cross-encoder" else self._llm(query, hits)
        except Exception as exc:  # failure recovery: degrade to first-stage order
            logger.warning("reranking failed (%s); using hybrid order", exc)
            return hits[:top_n]
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_n]

    def _cross_encoder(self, query: str, hits: list[Hit]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.s.reranker_model)
        return list(self._model.predict([(query, h.chunk.text) for h in hits]))

    def _llm(self, query: str, hits: list[Hit]) -> list[float]:
        from ..prompts import rerank_messages

        scores = [0.0] * len(hits)
        for start in range(0, len(hits), 10):
            batch = hits[start : start + 10]
            texts = [h.chunk.text for h in batch]
            res = self.llm.structured(rerank_messages(query, texts), RerankResult, task="rerank", context={"query": query, "passages": texts})
            for ps in res.scores:
                if 0 <= ps.index < len(batch):
                    scores[start + ps.index] = ps.score
        return scores
