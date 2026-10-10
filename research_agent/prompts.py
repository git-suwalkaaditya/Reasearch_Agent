"""All prompts in one place, so you can read and tune them easily.

Each function returns a list of chat messages. Structured tasks get their JSON schema
appended automatically by ``LLMClient.structured``.
"""
from __future__ import annotations

from .utils import truncate

GROUNDING_RULES = (
    "Ground every statement in the provided material. Never invent papers, numbers, authors or results. "
    "If the material does not support something, say so instead of guessing."
)


def _sys(text: str) -> dict:
    return {"role": "system", "content": text}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


# --------------------------------------------------------------------------- planning
def plan_messages(question: str, memory: str) -> list[dict]:
    return [
        _sys(
            "You are a senior research scientist planning a literature review. Break the research question into "
            "2-5 focused, answerable sub-questions (ids SQ1, SQ2, ...) and write 3-6 diverse academic search queries "
            "(short keyword phrases that would match paper titles/abstracts, mixing synonyms and specific method names). "
            "Also list key concepts and one sentence of inclusion criteria."
        ),
        _user(f"Research question: {question}\n\nWhat we already know from earlier work:\n{memory}"),
    ]


# --------------------------------------------------------------------------- search agent (tool calling)
def search_agent_messages(question: str, sub_questions: list[dict], queries: list[str], memory: str, n_found: int) -> list[dict]:
    sq = "\n".join(f"- {s['id']}: {s['question']}" for s in sub_questions)
    q = "\n".join(f"- {x}" for x in queries)
    return [
        _sys(
            "You are a literature-search agent. Use the tools to find papers relevant to the research question. "
            "Strategy: first check the local library, then run the suggested queries on the academic search tools, "
            "then refine queries if results look off-topic or thin (try synonyms, method names, survey/benchmark terms). "
            "Prefer several focused queries over one broad one. Call `finish` when you have enough candidates "
            "(roughly 15-40) or when further searching is unlikely to help."
        ),
        _user(
            f"Research question: {question}\nSub-questions:\n{sq}\n\nSuggested queries:\n{q}\n\n"
            f"Candidates found so far: {n_found}\nScratchpad:\n{memory}"
        ),
    ]


# --------------------------------------------------------------------------- screening
def screen_messages(question: str, sub_questions: list[dict], criteria: str, papers: list[dict]) -> list[dict]:
    sq = "\n".join(f"- {s['id']}: {s['question']}" for s in sub_questions)
    listing = "\n\n".join(
        f"paper_id: {p['paper_id']}\ntitle: {p['title']}\nyear: {p.get('year')}\nabstract: {truncate(p.get('abstract', ''), 900)}"
        for p in papers
    )
    return [
        _sys(
            "You screen papers for a literature review. For EVERY paper listed, give a relevance score 0-10 "
            "(10 = directly answers a sub-question with empirical evidence; 0 = unrelated), a one-line reason, and the "
            "ids of the sub-questions it helps answer. Judge only from the title and abstract."
        ),
        _user(f"Research question: {question}\nSub-questions:\n{sq}\nInclusion criteria: {criteria}\n\nPapers:\n{listing}"),
    ]


# --------------------------------------------------------------------------- extraction
def extract_messages(question: str, sub_question: dict, passages: list[dict]) -> list[dict]:
    block = "\n\n".join(
        f"[{p['chunk_id']}] ({p['ref']}, section: {p['section']})\n{p['text']}" for p in passages
    )
    return [
        _sys(
            "You extract evidence for a literature review. From the passages, extract findings that help answer the "
            "sub-question: concrete results (with numbers when present), methods, datasets, and stated limitations. "
            "Rules: each finding must come from ONE passage; set chunk_id to that passage id exactly as shown in "
            "brackets; `evidence` must be copied VERBATIM from that passage (max 40 words); `claim` restates it in your "
            "own words as a self-contained sentence. Return an empty list if nothing is relevant. " + GROUNDING_RULES
        ),
        _user(f"Overall question: {question}\nSub-question {sub_question['id']}: {sub_question['question']}\n\nPassages:\n{block}"),
    ]


# --------------------------------------------------------------------------- verification
def verify_messages(items: list[dict]) -> list[dict]:
    block = "\n\n".join(
        f"finding_id: {it['finding_id']}\nclaim: {it['claim']}\npassage: {truncate(it['passage'], 1400)}" for it in items
    )
    return [
        _sys(
            "You are a strict fact-checker. For each claim, decide whether the passage supports it: "
            "'supported' (fully entailed), 'partially_supported' (main point holds but a detail is overstated or "
            "missing), or 'unsupported' (not entailed, contradicted, or numbers differ). Give a brief rationale. "
            "Return a verdict for every finding_id."
        ),
        _user(block),
    ]


# --------------------------------------------------------------------------- reflection
def reflect_messages(question: str, sub_questions: list[dict], coverage: list[dict], findings_digest: str, memory: str, iteration: int, max_iter: int) -> list[dict]:
    cov = "\n".join(
        f"- {c['id']} ({c['question']}): {c['n_findings']} verified findings from {c['n_papers']} papers" for c in coverage
    )
    return [
        _sys(
            "You are reviewing the progress of a literature review. Decide if the verified evidence is sufficient to "
            "answer every sub-question with comparisons across at least two papers. If not, name the missing aspects "
            "and propose up to 4 NEW search queries (different from ones already tried) that target them. Be pragmatic: "
            "if evidence is reasonable, mark sufficient=true."
        ),
        _user(
            f"Question: {question}\nIteration {iteration} of max {max_iter}.\nCoverage:\n{cov}\n\n"
            f"Verified findings (digest):\n{findings_digest}\n\nScratchpad:\n{memory}"
        ),
    ]


# --------------------------------------------------------------------------- synthesis
def synthesize_messages(question: str, sub_questions: list[dict], findings: list[dict], papers: list[dict]) -> list[dict]:
    fl = "\n".join(f"{f['finding_id']} [{f['paper_ref']}] ({f['sub_question_id']}, {f['finding_type']}): {f['claim']}" for f in findings)
    pl = "\n".join(f"{p['paper_id']}: {p['title']} ({p.get('year')})" for p in papers)
    return [
        _sys(
            "You synthesise verified findings into a comparative analysis. Identify (1) agreements - conclusions "
            "supported by multiple papers, (2) disagreements or tensions - conflicting results, trade-offs, or "
            "different conditions, (3) a method comparison row per paper (approach, data/setting, key result), and "
            "(4) research gaps - things the evidence shows are missing, untested, or contradictory, each with why it "
            "matters and a concrete suggested direction. Reference findings only by the given finding ids. " + GROUNDING_RULES
        ),
        _user(f"Question: {question}\n\nPapers:\n{pl}\n\nVerified findings:\n{fl}"),
    ]


# --------------------------------------------------------------------------- writing
def write_messages(question: str, sub_questions: list[dict], findings: list[dict], synthesis: dict, feedback: str) -> list[dict]:
    sq = "\n".join(f"- {s['id']}: {s['question']}" for s in sub_questions)
    fl = "\n".join(f"[{f['cite']}] {f['claim']} (finding {f['finding_id']}, {f['sub_question_id']})" for f in findings)
    agree = "\n".join(f"- {a['statement']} ({', '.join(a['finding_ids'])})" for a in synthesis.get("agreements", []))
    disagree = "\n".join(f"- {a['statement']} ({', '.join(a['finding_ids'])})" for a in synthesis.get("disagreements", []))
    gaps = "\n".join(f"- {g['gap']} - {g.get('why_it_matters', '')} ({', '.join(g.get('finding_ids', []))})" for g in synthesis.get("gaps", []))
    fb = f"\n\nFix these problems from your previous draft:\n{feedback}" if feedback else ""
    return [
        _sys(
            "You write rigorous, citation-backed research reports in Markdown. Use ONLY the numbered findings provided. "
            "Cite with square-bracket numbers exactly as given, e.g. [1] or [2][3], placed right after the claim they "
            "support. Every sentence that states a fact or result must carry at least one citation. Do not cite numbers "
            "that are not in the list, and do not add a reference list (it is appended automatically).\n"
            "Structure:\n## Executive summary\n## Findings\n(one ### subsection per sub-question)\n## Comparison across studies\n"
            "## Research gaps and future directions\n## Limitations of this review\n" + GROUNDING_RULES
        ),
        _user(
            f"Research question: {question}\nSub-questions:\n{sq}\n\nNumbered findings:\n{fl}\n\n"
            f"Agreements:\n{agree or '-'}\n\nDisagreements:\n{disagree or '-'}\n\nGaps:\n{gaps or '-'}{fb}"
        ),
    ]


# --------------------------------------------------------------------------- reranking / memory / judge
def rerank_messages(query: str, passages: list[str]) -> list[dict]:
    block = "\n\n".join(f"[{i}] {truncate(p, 600)}" for i, p in enumerate(passages))
    return [
        _sys("Score how useful each passage is for answering the query, 0 (useless) to 10 (directly answers it). Score every index."),
        _user(f"Query: {query}\n\nPassages:\n{block}"),
    ]


def memory_summary_messages(summary: str, notes: list[str]) -> list[dict]:
    return [
        _sys("Condense the agent's scratchpad into a short summary (<= 120 words) keeping queries tried, key sources found, and failures."),
        _user(f"Existing summary: {summary or '-'}\nNotes:\n" + "\n".join(f"- {n}" for n in notes)),
    ]


def judge_messages(question: str, report: str, evidence: str) -> list[dict]:
    return [
        _sys(
            "You are an expert reviewer grading an AI-written literature review. Score 1-5 on faithfulness, coverage, "
            "synthesis, gap_quality and citation_quality using the evidence list as ground truth. Be critical; 5 means "
            "publication-quality. Justify briefly."
        ),
        _user(f"Question: {question}\n\nEvidence the report may use:\n{evidence}\n\nReport:\n{report}"),
    ]
