"""Deterministic offline LLM used for demos and tests (``--offline``).

It does not understand language; it imitates each task with simple lexical heuristics
using the ``context`` objects that the nodes pass alongside the prompt. This lets the
whole agent graph - tool calls, loops, verification, citation checks, evaluation -
run end-to-end with no API key and no network, which is ideal for learning and for
unit tests. Real runs use ``llm.OpenAICompatLLM`` and ignore ``context``.
"""
from __future__ import annotations

import re
from typing import Optional, Type

from pydantic import BaseModel

from .llm import BaseLLM, M
from .utils import token_overlap, tokenize, truncate

_LIMIT = re.compile(r"limit|however|fail|cannot|future|open question|further study|unknown|not evaluated|not studied|does not|did not|conflicts", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.split()) >= 6]


class MockLLM(BaseLLM):
    name = "mock"

    def __init__(self, bad_first_draft: bool = False) -> None:
        super().__init__()
        self.bad_first_draft = bad_first_draft
        self._drafts = 0

    # ------------------------------------------------------------------ API
    def complete(self, messages: list[dict], task: str = "", context: Optional[dict] = None, **kw) -> str:
        self.usage["calls"] += 1
        if task == "write":
            return self._write(context or {})
        return "OK"

    def structured(self, messages, schema: Type[M], task: str = "", context: Optional[dict] = None, max_repairs: int = 2) -> M:
        self.usage["calls"] += 1
        handler = getattr(self, f"_{task}", None)
        if handler is None:
            raise ValueError(f"MockLLM has no handler for task '{task}'")
        return schema.model_validate(handler(context or {}))

    def chat_with_tools(self, messages, tools, task: str = "", context: Optional[dict] = None) -> dict:
        self.usage["calls"] += 1
        ctx = context or {}
        tried = set(ctx.get("tried", []))
        tool_names = {t["function"]["name"] for t in tools}
        calls = []
        if ctx.get("step", 0) == 0 and "search_library" in tool_names:
            calls.append(("search_library", {"query": ctx["queries"][0]}))
        for q in ctx.get("queries", []):
            if q not in tried:
                for name in ("search_arxiv", "search_semantic_scholar"):
                    if name in tool_names:
                        calls.append((name, {"query": q, "max_results": 8}))
                        break
        if not calls:
            calls.append(("finish", {"reason": "all suggested queries searched"}))
        return {
            "content": None,
            "tool_calls": [{"id": f"call_{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)],
        }

    # ------------------------------------------------------------------ task handlers
    def _plan(self, ctx: dict) -> dict:
        q = ctx["question"].rstrip("?")
        topic = re.sub(r"^(how|what|which|why|when|where|does|do|is|are|can)\b(\s+(do|does|is|are|can))?\s*", "", q, flags=re.I)
        toks = tokenize(q)
        core = " ".join(toks[:6])
        return {
            "sub_questions": [
                {"id": "SQ1", "question": f"Which methods and system designs address how {topic}?"},
                {"id": "SQ2", "question": f"What accuracy, privacy and efficiency results are reported on how {topic}?"},
                {"id": "SQ3", "question": f"What limitations and open problems remain on how {topic}?"},
            ],
            "search_queries": [core, " ".join(toks[:3]) + " privacy accuracy", " ".join(toks[:3]) + " limitations"],
            "key_concepts": toks[:6],
            "inclusion_criteria": "Empirical or methodological papers on the topic.",
        }

    def _screen(self, ctx: dict) -> dict:
        q = ctx["question"]
        out = []
        for p in ctx["papers"]:
            ov = token_overlap(q, p["title"] + " " + p.get("abstract", ""))
            out.append({"paper_id": p["paper_id"], "relevance": min(10, round(ov * 22)), "reason": "lexical overlap with question", "sub_question_ids": ["SQ1", "SQ2", "SQ3"]})
        return {"decisions": out}

    def _extract(self, ctx: dict) -> dict:
        sq = ctx["sub_question"]["question"] + " " + ctx.get("question", "")
        findings = []
        for p in ctx["passages"]:
            sents = _sentences(p["text"])
            if not sents:
                continue
            wants_limits = bool(re.search(r"limitation|open problem", sq, re.I))
            best = max(sents, key=lambda s: token_overlap(sq, s) + (0.3 if re.search(r"\d", s) else 0)
                       + (0.6 if wants_limits and _LIMIT.search(s) else 0))
            if token_overlap(sq, best) < 0.12:
                continue
            words = best.split()
            evidence = " ".join(words[:40])
            ftype = "limitation" if _LIMIT.search(best) else ("result" if re.search(r"\d", best) else "method")
            findings.append({"claim": truncate(best, 300), "chunk_id": p["chunk_id"], "evidence": evidence, "finding_type": ftype})
            if len(findings) >= 4:
                break
        return {"findings": findings}

    def _verify(self, ctx: dict) -> dict:
        out = []
        for it in ctx["items"]:
            ov = token_overlap(it["claim"], it["passage"])
            verdict = "supported" if ov >= 0.8 else "partially_supported" if ov >= 0.5 else "unsupported"
            out.append({"finding_id": it["finding_id"], "verdict": verdict, "rationale": f"word overlap {ov:.2f}"})
        return {"verdicts": out}

    def _reflect(self, ctx: dict) -> dict:
        missing = [c for c in ctx["coverage"] if c["n_findings"] == 0 or c["n_papers"] < 3]
        if not missing or ctx.get("iteration", 1) >= ctx.get("max_iter", 2):
            return {"sufficient": True, "coverage_notes": "All sub-questions have evidence."}
        return {
            "sufficient": False,
            "coverage_notes": "Some sub-questions lack multi-paper evidence.",
            "missing_aspects": [c["question"] for c in missing],
            "new_queries": [" ".join(tokenize(c["question"])[:5]) for c in missing][:4],
        }

    def _synthesize(self, ctx: dict) -> dict:
        findings, papers = ctx["findings"], ctx["papers"]
        by_sq: dict[str, list[dict]] = {}
        for f in findings:
            by_sq.setdefault(f["sub_question_id"], []).append(f)
        agreements = []
        for sq, fs in by_sq.items():
            if len({f["paper_id"] for f in fs}) >= 2:
                agreements.append({"statement": f"Several studies provide evidence on {sq}: " + truncate(fs[0]["claim"], 120), "finding_ids": [f["finding_id"] for f in fs[:3]]})
        results = [f for f in findings if f["finding_type"] == "result"]
        disagreements = []
        if len({f["paper_id"] for f in results}) >= 2:
            a, b = results[0], next(f for f in results if f["paper_id"] != results[0]["paper_id"])
            disagreements.append({"statement": "Reported results differ across settings: " + truncate(a["claim"], 90) + " vs. " + truncate(b["claim"], 90), "finding_ids": [a["finding_id"], b["finding_id"]]})
        rows = []
        for p in papers:
            pf = [f for f in findings if f["paper_id"] == p["paper_id"]]
            if not pf:
                continue
            method = next((f for f in pf if f["finding_type"] == "method"), pf[0])
            result = next((f for f in pf if f["finding_type"] == "result"), pf[-1])
            rows.append({"paper_id": p["paper_id"], "approach": truncate(method["claim"], 110), "data_or_setting": "see paper", "key_result": truncate(result["claim"], 110)})
        gaps, seen_claims = [], set()
        for f in findings:
            if f["finding_type"] == "limitation" and f["claim"] not in seen_claims:
                seen_claims.add(f["claim"])
                gaps.append({"gap": truncate(f["claim"], 160), "why_it_matters": "Reported as an open limitation.", "finding_ids": [f["finding_id"]],
                             "suggested_direction": "Design experiments that address this limitation directly."})
        gaps = gaps[:4]
        return {"agreements": agreements, "disagreements": disagreements, "method_comparison": rows, "gaps": gaps}

    def _write(self, ctx: dict) -> str:
        self._drafts += 1
        findings, sqs, syn = ctx["findings"], ctx["sub_questions"], ctx["synthesis"]
        lines = ["## Executive summary", ""]
        top = findings[:3]
        lines.append(" ".join(f"{truncate(f['claim'], 160)} [{f['cite']}]" for f in top))
        if self.bad_first_draft and self._drafts == 1:
            lines.append("This field is growing rapidly [99].")  # deliberately invalid citation (tests the repair loop)
        lines += ["", "## Findings", ""]
        for sq in sqs:
            lines += [f"### {sq['id']}: {sq['question']}", ""]
            fs = [f for f in findings if f["sub_question_id"] == sq["id"]]
            if not fs:
                lines.append("The retrieved evidence did not address this sub-question.")
            for f in fs[:5]:
                lines.append(f"- {f['claim']} [{f['cite']}]")
            lines.append("")
        cite_of = {f["finding_id"]: f["cite"] for f in findings}
        lines += ["## Comparison across studies", ""]
        for a in syn.get("agreements", []) + syn.get("disagreements", []):
            cites = "".join(f"[{n}]" for n in sorted({cite_of[i] for i in a["finding_ids"] if i in cite_of}))
            lines.append(f"- {a['statement']} {cites}")
        lines += ["", "## Research gaps and future directions", ""]
        for g in syn.get("gaps", []):
            cites = "".join(f"[{n}]" for n in sorted({cite_of[i] for i in g.get("finding_ids", []) if i in cite_of}))
            lines.append(f"- **{truncate(g['gap'], 140)}** {g.get('suggested_direction', '')} {cites}")
        lines += ["", "## Limitations of this review", "", "This report was generated in offline demo mode from bundled demo papers."]
        return "\n".join(lines)

    def _rerank(self, ctx: dict) -> dict:
        return {"scores": [{"index": i, "score": round(min(10, token_overlap(ctx["query"], p) * 12), 2)} for i, p in enumerate(ctx["passages"])]}

    def _memory_summary(self, ctx: dict) -> dict:
        return {"summary": truncate((ctx.get("summary", "") + " " + " ".join(ctx.get("notes", []))).strip(), 600)}

    def _judge(self, ctx: dict) -> dict:
        m = ctx.get("metrics", {})
        s = lambda x: max(1, min(5, round(1 + 4 * x)))  # noqa: E731
        return {
            "faithfulness": s(m.get("claim_support_rate", 0.5)),
            "coverage": s(m.get("sub_question_coverage", 0.5)),
            "synthesis": 3,
            "gap_quality": 3 if m.get("n_gaps", 0) else 1,
            "citation_quality": s(m.get("citation_validity", 0.5)),
            "justification": "Heuristic mock judge (offline mode).",
        }
