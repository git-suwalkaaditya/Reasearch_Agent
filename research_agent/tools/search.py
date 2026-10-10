"""Paper search tools exposed to the LLM via tool calling.

* ``search_arxiv``            - arXiv API (Atom XML), no key needed.
* ``search_semantic_scholar`` - Semantic Scholar Graph API (key optional, raises rate limits).
* ``search_library``          - the agent's own persistent paper library (long-term memory).
* ``finish``                  - lets the model end the search loop.

In offline mode the two web tools search the bundled demo papers instead.
"""
from __future__ import annotations

import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import requests

from ..config import Settings
from ..memory import PersistentStore
from ..retrieval.bm25 import BM25
from ..schemas import Paper
from ..utils import logger, retry, tokenize, truncate

DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_papers.json"
USER_AGENT = "research-agent/1.0 (educational project)"


def load_demo_papers() -> tuple[list[Paper], dict[str, list[dict]]]:
    raw = json.loads(DEMO_PATH.read_text(encoding="utf-8"))["papers"]
    papers = [Paper(**{k: v for k, v in p.items() if k != "sections"}, source="demo") for p in raw]
    sections = {p["paper_id"]: p.get("sections", []) for p in raw}
    return papers, sections


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.time()
            if delay > 0:
                time.sleep(delay)
            self._last = time.time()


def _bm25_rank(query: str, papers: list[Paper], k: int) -> list[Paper]:
    if not papers:
        return []
    bm = BM25().fit([tokenize(p.title + " " + p.title + " " + p.abstract) for p in papers])
    scores = bm.scores(tokenize(query))
    order = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
    return [papers[i] for i in order[:k] if scores[i] > 0]


class SearchTools:
    ARXIV_URL = "http://export.arxiv.org/api/query"
    S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    S2_FIELDS = "title,abstract,year,authors,externalIds,openAccessPdf,citationCount,url,venue"

    def __init__(self, settings: Settings, store: PersistentStore) -> None:
        self.s = settings
        self.store = store
        self._arxiv_rl = _RateLimiter(3.1)   # arXiv asks for >= 3 s between requests
        self._s2_rl = _RateLimiter(1.1 if settings.semantic_scholar_api_key else 3.0)
        self._demo: list[Paper] | None = None
        if settings.offline:
            self._demo, _ = load_demo_papers()

    # ------------------------------------------------------------------ tool specs
    def tool_specs(self) -> list[dict]:
        def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}

        q = {"query": {"type": "string", "description": "Short keyword query, e.g. 'federated gradient boosting differential privacy'"}}
        n = {"max_results": {"type": "integer", "description": "1-20", "default": 8}}
        specs = [fn("search_library", "Search papers the agent has already stored in its local library (from earlier runs).", q, ["query"])]
        if "arxiv" in self.s.sources:
            specs.append(fn("search_arxiv", "Search arXiv for papers (title/abstract/full metadata).", {**q, **n}, ["query"]))
        if "semantic_scholar" in self.s.sources:
            specs.append(fn("search_semantic_scholar", "Search Semantic Scholar across all fields; returns citation counts.", {**q, **n}, ["query"]))
        specs.append(fn("finish", "Stop searching when you have enough relevant candidates.", {"reason": {"type": "string"}}, ["reason"]))
        return specs

    def execute(self, name: str, args: dict) -> tuple[list[Paper], str]:
        """Run a tool call and return (papers, observation text for the LLM)."""
        dispatch: dict[str, Callable[..., list[Paper]]] = {
            "search_library": self.search_library,
            "search_arxiv": self.search_arxiv,
            "search_semantic_scholar": self.search_semantic_scholar,
        }
        if name not in dispatch:
            return [], f"Unknown tool '{name}'."
        query = str(args.get("query", "")).strip()
        if not query:
            return [], "Error: empty query."
        k = max(1, min(20, int(args.get("max_results", self.s.results_per_query) or self.s.results_per_query)))
        try:
            papers = dispatch[name](query, k)
        except Exception as exc:  # tool failure is reported back to the model, not raised
            logger.warning("%s(%r) failed: %s", name, query, exc)
            return [], f"Tool error from {name}: {type(exc).__name__}: {truncate(str(exc), 200)}. Try another tool or query."
        if not papers:
            return [], f"{name}('{query}') returned no results."
        lines = [f"- {p.paper_id} | {truncate(p.title, 110)} ({p.year})" for p in papers]
        return papers, f"{name}('{query}') returned {len(papers)} papers:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ tools
    def search_library(self, query: str, k: int = 8) -> list[Paper]:
        return _bm25_rank(query, self.store.all_papers(), k)

    def search_arxiv(self, query: str, k: int = 8) -> list[Paper]:
        if self.s.offline:
            return _bm25_rank(query, self._demo or [], k)
        terms = [t for t in tokenize(query) if len(t) > 2][:5] or [query]
        strict = " AND ".join(f'all:"{t}"' for t in terms)
        papers = self._arxiv_query(strict, k)
        if len(papers) < 3 and len(terms) > 1:  # too strict -> relax to OR (failure recovery)
            papers += [p for p in self._arxiv_query(" OR ".join(f'all:"{t}"' for t in terms), k) if p.paper_id not in {x.paper_id for x in papers}]
        return papers[:k]

    def _arxiv_query(self, search_query: str, k: int) -> list[Paper]:
        params = {"search_query": search_query, "start": 0, "max_results": k, "sortBy": "relevance"}

        def call() -> str:
            self._arxiv_rl.wait()
            r = requests.get(self.ARXIV_URL, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r.text

        xml = retry(call, attempts=3, base_delay=3, exceptions=(requests.RequestException,), what="arXiv search")
        return parse_arxiv_atom(xml)

    def search_semantic_scholar(self, query: str, k: int = 8) -> list[Paper]:
        if self.s.offline:
            return _bm25_rank(query, self._demo or [], k)
        headers = {"User-Agent": USER_AGENT}
        if self.s.semantic_scholar_api_key:
            headers["x-api-key"] = self.s.semantic_scholar_api_key
        params = {"query": query, "limit": k, "fields": self.S2_FIELDS}

        def call() -> dict:
            self._s2_rl.wait()
            r = requests.get(self.S2_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                raise requests.RequestException("rate limited (429)")
            r.raise_for_status()
            return r.json()

        data = retry(call, attempts=4, base_delay=3, exceptions=(requests.RequestException,), what="Semantic Scholar search")
        return parse_semantic_scholar(data)


# ---------------------------------------------------------------------- parsers (pure, testable)
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse_arxiv_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", _ATOM):
        raw_id = (e.findtext("a:id", default="", namespaces=_ATOM) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id.strip())
        if not arxiv_id:
            continue
        pdf = next((l.get("href", "") for l in e.findall("a:link", _ATOM) if l.get("title") == "pdf"), "")
        published = e.findtext("a:published", default="", namespaces=_ATOM)
        papers.append(
            Paper(
                paper_id=f"arxiv:{arxiv_id}",
                arxiv_id=arxiv_id,
                title=re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=_ATOM)).strip(),
                abstract=re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=_ATOM)).strip(),
                authors=[a.findtext("a:name", default="", namespaces=_ATOM) for a in e.findall("a:author", _ATOM)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf or f"https://arxiv.org/pdf/{arxiv_id}",
                venue=e.findtext("arxiv:journal_ref", default="", namespaces=_ATOM) or "arXiv",
                source="arxiv",
            )
        )
    return papers


def parse_semantic_scholar(data: dict) -> list[Paper]:
    papers = []
    for d in data.get("data") or []:
        if not d.get("title"):
            continue
        ext = d.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv", "") or ""
        pid = f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{d.get('paperId', '')}"
        pdf = (d.get("openAccessPdf") or {}).get("url", "") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "")
        papers.append(
            Paper(
                paper_id=pid,
                arxiv_id=arxiv_id,
                title=d["title"].strip(),
                abstract=(d.get("abstract") or "").strip(),
                authors=[a.get("name", "") for a in d.get("authors") or []],
                year=d.get("year"),
                url=d.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                pdf_url=pdf,
                venue=d.get("venue") or "",
                citation_count=d.get("citationCount"),
                source="semantic_scholar",
            )
        )
    return papers
