"""Reading papers: download PDF -> extract text -> split into sections -> chunk.

Failure recovery is built in: if the PDF is missing, paywalled, too large or unparsable,
the paper is still used via its abstract (status ``abstract_only``) and the problem is
recorded, instead of crashing the run.
"""
from __future__ import annotations

import re
from pathlib import Path

import requests

from ..config import Settings
from ..memory import PersistentStore
from ..schemas import Chunk, Paper
from ..utils import logger, retry
from .search import USER_AGENT, load_demo_papers

MAX_PDF_BYTES = 25 * 1024 * 1024

_HEADINGS = (
    r"abstract|introduction|related work|background|preliminaries|problem (?:setup|formulation|statement)|"
    r"methods?|methodology|approach|proposed (?:method|approach|framework)|model|framework|system design|"
    r"experiments?|experimental (?:setup|results|evaluation)|evaluation|results(?: and discussion)?|discussion|"
    r"analysis|ablation(?: study| studies)?|limitations?|threats to validity|conclusions?(?: and future work)?|"
    r"future work|references|bibliography|acknowledg(?:e)?ments?|appendix"
)
HEADING_RE = re.compile(rf"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+|[A-H])\.?\s+)?({_HEADINGS})\b[\s:.]*$", re.IGNORECASE)
_STOP_SECTIONS = {"references", "bibliography", "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement"}


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split raw paper text into (section, text) pairs using heading heuristics."""
    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    for line in text.splitlines():
        stripped = line.strip()
        m = HEADING_RE.match(stripped) if 3 <= len(stripped) <= 70 else None
        if m:
            name = m.group(1).strip().title()
            if name.lower() in _STOP_SECTIONS:
                break  # everything after references is noise for retrieval
            sections.append((name, []))
        else:
            sections[-1][1].append(line)
    out = []
    for name, lines in sections:
        body = re.sub(r"-\n(\w)", r"\1", "\n".join(lines))   # join hyphenated words
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) > 80:
            out.append((name, body))
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def chunk_sections(paper_id: str, sections: list[tuple[str, str]], size: int = 1000, overlap: int = 150, start: int = 0) -> list[Chunk]:
    """Sentence-aware chunking with overlap, never crossing section boundaries."""
    chunks: list[Chunk] = []
    pos = start
    for section, body in sections:
        sentences = _SENT_SPLIT.split(body)
        buf = ""
        for sent in sentences:
            if len(sent) > size:  # pathological long "sentence" (tables, equations)
                sent = sent[:size]
            if buf and len(buf) + len(sent) + 1 > size:
                chunks.append(Chunk(chunk_id=f"{paper_id}#{pos}", paper_id=paper_id, section=section, text=buf.strip(), position=pos))
                pos += 1
                tail = buf[-overlap:]
                buf = (tail[tail.find(" ") + 1 :] if " " in tail else "") + " " + sent
            else:
                buf = f"{buf} {sent}" if buf else sent
        if buf.strip():
            chunks.append(Chunk(chunk_id=f"{paper_id}#{pos}", paper_id=paper_id, section=section, text=buf.strip(), position=pos))
            pos += 1
    return chunks


class PaperReader:
    def __init__(self, settings: Settings, store: PersistentStore) -> None:
        self.s = settings
        self.store = store
        self._demo_sections: dict[str, list[dict]] = {}
        if settings.offline:
            _, self._demo_sections = load_demo_papers()

    def read(self, paper: Paper) -> tuple[list[Chunk], str]:
        """Return (chunks, status). status: cached | fulltext | demo | abstract_only:<reason>."""
        if self.store.has_fulltext(paper.paper_id):
            return self.store.get_chunks(paper.paper_id), "cached"

        abstract = [Chunk(chunk_id=f"{paper.paper_id}#0", paper_id=paper.paper_id, section="Abstract", text=paper.abstract, position=0)] if paper.abstract else []

        if paper.paper_id in self._demo_sections:
            secs = [(s["section"], s["text"]) for s in self._demo_sections[paper.paper_id]]
            chunks = abstract + chunk_sections(paper.paper_id, secs, self.s.chunk_chars, self.s.chunk_overlap, start=1)
            self.store.save_chunks(chunks)
            return chunks, "demo"

        status = "fulltext"
        body: list[Chunk] = []
        try:
            text = self._download_and_extract(paper)
            sections = [s for s in split_sections(text) if s[0] != "Abstract"]
            body = chunk_sections(paper.paper_id, sections, self.s.chunk_chars, self.s.chunk_overlap, start=1)
            if not body:
                raise ValueError("no text extracted")
        except Exception as exc:
            status = f"abstract_only: {type(exc).__name__}: {str(exc)[:120]}"
            logger.warning("Using abstract only for %s (%s)", paper.paper_id, status)
        chunks = abstract + body
        if chunks:
            self.store.save_chunks(chunks)
        return chunks, status

    def _download_and_extract(self, paper: Paper) -> str:
        url = paper.pdf_url or (f"https://arxiv.org/pdf/{paper.arxiv_id}" if paper.arxiv_id else "")
        if not url:
            raise ValueError("no PDF url")
        path = self.s.pdf_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", paper.paper_id) + ".pdf")
        if not path.exists():
            def call() -> bytes:
                r = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT}, stream=True)
                r.raise_for_status()
                data = b""
                for block in r.iter_content(65536):
                    data += block
                    if len(data) > MAX_PDF_BYTES:
                        raise ValueError("PDF larger than 25 MB")
                return data

            data = retry(call, attempts=2, base_delay=3, exceptions=(requests.RequestException,), what=f"download {paper.paper_id}")
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF (paywall or HTML page?)")
            path.write_bytes(data)
        return extract_pdf_text(path, self.s.max_pdf_pages)


def extract_pdf_text(path: Path, max_pages: int = 30) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # a single broken page should not lose the paper
            continue
    return "\n".join(pages)
