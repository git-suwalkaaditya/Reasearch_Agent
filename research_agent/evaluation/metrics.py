"""Automatic metrics for agent quality, factuality and citation correctness.

All metrics are computed from the final agent state, so they work for any run.

Factuality
  claim_support_rate          verified claims / extracted claims (LLM fact-check + quote check)
  evidence_quote_match_rate   extracted claims whose verbatim quote really occurs in the cited passage
Citation correctness
  citation_validity           citations in the report that point to a real reference
  citation_support_rate       cited sentences whose content overlaps the cited paper's verified claims
  uncited_sentence_ratio      factual sentences with no citation at all
Coverage / process
  sub_question_coverage       sub-questions with >= 1 verified finding
  retrieval recall            share of known-relevant papers that the search found / selected
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from ..report import CITE_RE, expand_citation_group, split_sentences
from ..schemas import Finding
from ..utils import normalize_title, token_overlap, tokenize


def _report_body(state: dict) -> str:
    md = state.get("report_md") or state.get("report_draft") or ""
    return md.split("\n## References", 1)[0]


def citation_metrics(state: dict) -> dict:
    cmap = state.get("citation_map", {})
    valid = set(cmap.values())
    claims_by_num: dict[int, list[str]] = {}
    for f in state.get("findings", []):
        fd = Finding(**f)
        if fd.is_verified and fd.paper_id in cmap:
            claims_by_num.setdefault(cmap[fd.paper_id], []).append(fd.claim + " " + fd.evidence)

    total = invalid = cited_sents = supported_sents = factual = uncited = 0
    section = ""
    for line in _report_body(state).splitlines():
        if line.startswith("#"):
            section = line.lower()
            continue
        if line.startswith("*Generated") or "limitations of this review" in section or "method comparison" in section:
            continue
        for sent in split_sentences(line):
            nums = [n for g in CITE_RE.findall(sent) for n in expand_citation_group(g)]
            total += len(nums)
            invalid += sum(1 for n in nums if n not in valid)
            bare = CITE_RE.sub("", sent)
            if len(tokenize(bare)) < 5:
                continue
            factual += 1
            good = [n for n in nums if n in valid]
            if not nums:
                uncited += 1
            elif good:
                cited_sents += 1
                best = max((token_overlap(bare, c) for n in good for c in claims_by_num.get(n, [])), default=0.0)
                supported_sents += best >= 0.2
    return {
        "n_citations": total,
        "citation_validity": round((total - invalid) / total, 3) if total else 0.0,
        "citation_support_rate": round(supported_sents / cited_sents, 3) if cited_sents else 0.0,
        "uncited_sentence_ratio": round(uncited / factual, 3) if factual else 0.0,
    }


def compute_metrics(state: dict) -> dict:
    findings = [Finding(**f) for f in state.get("findings", [])]
    verified = [f for f in findings if f.is_verified]
    sqs = (state.get("plan") or {}).get("sub_questions", [])
    covered = {f.sub_question_id for f in verified}
    read = state.get("read_status", {})
    syn = state.get("synthesis") or {}
    m = {
        "status": state.get("status"),
        "iterations": state.get("iteration", 1),
        "n_candidates": len(state.get("candidates", {})),
        "n_selected": len(state.get("selected", [])),
        "n_read": len(read),
        "n_fulltext": sum(1 for s in read.values() if not s.startswith("abstract_only")),
        "n_findings": len(findings),
        "n_verified": len(verified),
        "verdicts": dict(Counter(f.verdict or "none" for f in findings)),
        "claim_support_rate": round(len(verified) / len(findings), 3) if findings else 0.0,
        "evidence_quote_match_rate": round(sum(1 for f in findings if f.quote_match) / len(findings), 3) if findings else 0.0,
        "sub_question_coverage": round(len(covered & {s["id"] for s in sqs}) / len(sqs), 3) if sqs else 0.0,
        "n_cited_papers": len(state.get("citation_map", {})),
        "n_agreements": len(syn.get("agreements", [])),
        "n_disagreements": len(syn.get("disagreements", [])),
        "n_gaps": len(syn.get("gaps", [])),
        "n_errors": len(state.get("errors", [])),
        "rewrites": max(0, state.get("write_attempts", 1) - 1),
        "runtime_s": round(sum(t.get("seconds", 0) for t in state.get("trace", [])), 1),
    }
    m.update(citation_metrics(state))
    return m


def _title_match(a: str, b: str) -> bool:
    a, b = normalize_title(a), normalize_title(b)
    return a == b or (min(len(a), len(b)) > 15 and (a in b or b in a)) or SequenceMatcher(None, a, b).ratio() > 0.9


def retrieval_recall(state: dict, gold_titles: list[str]) -> dict:
    """Did the agent find / read the papers a human expert would expect?"""
    if not gold_titles:
        return {}
    cand = [p["title"] for p in state.get("candidates", {}).values()]
    sel = [state["candidates"][pid]["title"] for pid in state.get("selected", []) if pid in state.get("candidates", {})]
    found = sum(1 for g in gold_titles if any(_title_match(g, t) for t in cand))
    chosen = sum(1 for g in gold_titles if any(_title_match(g, t) for t in sel))
    return {"recall_candidates": round(found / len(gold_titles), 3), "recall_selected": round(chosen / len(gold_titles), 3)}
