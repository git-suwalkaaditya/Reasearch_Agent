"""Report rendering. The LLM writes the prose; everything verifiable (reference list,
method table, evidence appendix) is rendered deterministically from the state, so
references can never be hallucinated."""
from __future__ import annotations

import datetime as dt
import re

from .schemas import Finding, Paper
from .utils import truncate

CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*_(])")
_TRAILING_CITES = re.compile(r"([.!?])\s*((?:\[\d[\d,\s\u2013-]*\]\s*)+)")


def expand_citation_group(group: str) -> list[int]:
    """'1, 3-5' -> [1, 3, 4, 5]"""
    out: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        rng = re.split(r"\s*[–-]\s*", part)
        if len(rng) == 2 and rng[0].isdigit() and rng[1].isdigit() and int(rng[1]) >= int(rng[0]) and int(rng[1]) - int(rng[0]) < 50:
            out += list(range(int(rng[0]), int(rng[1]) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def split_sentences(line: str) -> list[str]:
    line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line).strip()
    if not line or line.startswith("|"):
        return []
    # "claim. [1][2] Next" -> "claim [1][2]. Next" so citations stay with their sentence
    line = _TRAILING_CITES.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", line).strip()
    return [s for s in _SENT_RE.split(line) if s.strip()]


def format_reference(n: int, p: Paper) -> str:
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "") if p.authors else "Unknown authors"
    venue = f" *{p.venue}*." if p.venue else ""
    link = f" {p.url}" if p.url else ""
    return f"[{n}] {authors}. **{p.title}**.{venue} {p.year or 'n.d.'}.{link}"


def _md_cell(text: str) -> str:
    return truncate(str(text), 160).replace("|", "/").replace("\n", " ")


def render_report(state: dict, body: str) -> str:
    cmap: dict[str, int] = state.get("citation_map", {})
    papers = {pid: Paper(**state["candidates"][pid]) for pid in cmap}
    findings = [Finding(**f) for f in state.get("findings", [])]
    verified = [f for f in findings if f.is_verified]
    syn = state.get("synthesis", {}) or {}
    n_iter = state.get("iteration", 1)

    lines = [
        f"# Research report: {state['question']}",
        "",
        f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M} · run `{state['run_id']}` · {len(state.get('candidates', {}))} papers found, "
        f"{len(state.get('read_status', {}))} read, {len(cmap)} cited · {len(verified)}/{len(findings)} extracted claims verified · "
        f"{n_iter} search iteration(s)*",
        "",
        body.strip(),
        "",
    ]

    rows = syn.get("method_comparison", [])
    if rows:
        lines += ["## Method comparison", "", "| Study | Approach | Data / setting | Key result |", "|---|---|---|---|"]
        for r in rows:
            n = cmap.get(r["paper_id"])
            if n is None:
                continue
            lines.append(f"| [{n}] {_md_cell(papers[r['paper_id']].short_ref())} | {_md_cell(r['approach'])} | {_md_cell(r.get('data_or_setting', ''))} | {_md_cell(r.get('key_result', ''))} |")
        lines.append("")

    lines += ["## References", ""]
    for pid, n in sorted(cmap.items(), key=lambda kv: kv[1]):
        lines.append(format_reference(n, papers[pid]))
        lines.append("")

    if verified:
        lines += ["<details><summary>Evidence appendix: verified claims and the exact quotes they rest on</summary>", ""]
        for f in verified:
            sec = f.chunk_id
            lines.append(f"- **{f.finding_id}** [{cmap.get(f.paper_id, '?')}] ({f.sub_question_id}, {f.verdict}) {f.claim}  \n  > \"{truncate(f.evidence, 300)}\" — `{sec}`")
        lines += ["", "</details>", ""]
    return "\n".join(lines)
