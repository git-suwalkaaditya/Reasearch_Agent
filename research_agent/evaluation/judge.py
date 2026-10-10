"""LLM-as-judge: a rubric-based quality score for the final report.

Automatic metrics catch broken citations and unsupported claims; a judge model catches
things they can't (is it a real synthesis or just a list? are the gaps insightful?).
For a fair evaluation, use a different (ideally stronger) model as the judge than the
one that wrote the report.
"""
from __future__ import annotations

from ..prompts import judge_messages
from ..schemas import Finding, JudgeScores


def judge_report(llm, state: dict, metrics: dict) -> dict:
    cmap = state.get("citation_map", {})
    evidence = "\n".join(
        f"[{cmap.get(f.paper_id, '?')}] {f.claim} (quote: \"{f.evidence[:200]}\")"
        for f in (Finding(**x) for x in state.get("findings", []))
        if f.is_verified
    )[:12000]
    report = (state.get("report_md") or "")[:16000]
    res = llm.structured(judge_messages(state["question"], report, evidence or "(none)"), JudgeScores, task="judge",
                         context={"metrics": metrics})
    scores = res.model_dump()
    scores["judge_mean"] = round(sum(scores[k] for k in ("faithfulness", "coverage", "synthesis", "gap_quality", "citation_quality")) / 5, 2)
    return scores
