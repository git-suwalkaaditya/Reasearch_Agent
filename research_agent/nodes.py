"""The agent's nodes. Each takes the state dict and returns a partial update.

Workflow (see agent.py for the wiring):

    plan -> search -> screen -> read -> extract -> verify -> reflect --(gaps?)--> search ...
                                                               \\--(enough)--> synthesize -> write -> check_citations
                                                                                               ^            |
                                                                                               +--(errors)--+
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .config import Settings
from .llm import BaseLLM, ToolCallingUnsupported
from .memory import PersistentStore, WorkingMemory
from .prompts import (
    extract_messages,
    plan_messages,
    reflect_messages,
    screen_messages,
    search_agent_messages,
    synthesize_messages,
    verify_messages,
    write_messages,
)
from .report import CITE_RE, expand_citation_group, render_report, split_sentences
from .retrieval import HybridIndex, Reranker
from .schemas import (
    ExtractionResult,
    Finding,
    Paper,
    Reflection,
    ResearchPlan,
    ScreeningResult,
    Synthesis,
    VerificationResult,
)
from .tools import PaperReader, SearchTools
from .retrieval.bm25 import BM25
from .utils import chunked, fuzzy_contains, logger, normalize_title, token_overlap, tokenize, truncate

MIN_RELEVANCE = 6
MAX_WRITE_ATTEMPTS = 2


class Nodes:
    def __init__(self, settings: Settings, llm: BaseLLM, store: PersistentStore, tools: SearchTools,
                 reader: PaperReader, index: HybridIndex, reranker: Reranker) -> None:
        self.s, self.llm, self.store = settings, llm, store
        self.tools, self.reader, self.index, self.reranker = tools, reader, index, reranker

    # ======================================================================= helpers
    @staticmethod
    def _note(state: dict, text: str) -> dict:
        return WorkingMemory.add(state.get("memory") or WorkingMemory.empty(), text)

    def _paper(self, state: dict, pid: str) -> Paper:
        return Paper(**state["candidates"][pid])

    def _ensure_indexed(self, state: dict) -> None:
        """After a resume the in-memory index is empty; rebuild it from the persistent store."""
        for pid in state.get("read_status", {}):
            chunks = self.store.get_chunks(pid)
            if chunks:
                self.index.add(chunks, title=state["candidates"].get(pid, {}).get("title", ""))

    @staticmethod
    def _findings(state: dict) -> list[Finding]:
        return [Finding(**f) for f in state.get("findings", [])]

    # ======================================================================= 1. plan
    def plan(self, state: dict) -> dict:
        memory = WorkingMemory.render(state.get("memory", {}))
        plan = self.llm.structured(plan_messages(state["question"], memory), ResearchPlan, task="plan",
                                   context={"question": state["question"]})
        mem = self._note(state, f"Planned {len(plan.sub_questions)} sub-questions; initial queries: {plan.search_queries}")
        return {
            "plan": plan.model_dump(),
            "pending_queries": plan.search_queries,
            "memory": mem,
            "_summary": f"{len(plan.sub_questions)} sub-questions, {len(plan.search_queries)} queries",
        }

    # ======================================================================= 2. search (tool calling)
    def search(self, state: dict) -> dict:
        plan = state["plan"]
        queries = [q for q in (state.get("pending_queries") or plan["search_queries"])]
        candidates = dict(state.get("candidates", {}))
        tried = list(state.get("queries_tried", []))
        memory = state.get("memory") or WorkingMemory.empty()
        titles = {normalize_title(p["title"]): pid for pid, p in candidates.items()}
        new_ids: list[str] = []

        def add(papers: list[Paper]) -> None:
            for p in papers:
                key_t = normalize_title(p.title)
                if p.paper_id in candidates or key_t in titles:
                    continue
                candidates[p.paper_id] = p.model_dump()
                titles[key_t] = p.paper_id
                self.store.upsert_paper(p)
                new_ids.append(p.paper_id)

        def run_tool(name: str, args: dict) -> str:
            nonlocal memory
            papers, obs = self.tools.execute(name, args)
            add(papers)
            q = str(args.get("query", ""))
            if name != "search_library" and q and q not in tried:
                tried.append(q)
            memory = WorkingMemory.add(memory, f"{name}('{q}') -> {len(papers)} results")
            return obs

        used_tools = False
        if self.llm.supports_tools:
            try:
                messages = search_agent_messages(state["question"], plan["sub_questions"], queries,
                                                 WorkingMemory.render(memory), len(candidates))
                specs = self.tools.tool_specs()
                for step in range(self.s.max_tool_steps):
                    resp = self.llm.chat_with_tools(messages, specs, task="search_agent",
                                                    context={"queries": queries, "tried": tried, "step": step})
                    calls = resp["tool_calls"]
                    if not calls:
                        break
                    used_tools = True
                    messages.append(resp.get("raw_message") or {
                        "role": "assistant", "content": resp.get("content"),
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls],
                    })
                    finished = False
                    for c in calls:
                        if c["name"] == "finish":
                            finished, obs = True, "Search finished."
                        else:
                            obs = run_tool(c["name"], c["arguments"])
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
                    if finished or len(candidates) >= 80:
                        break
            except ToolCallingUnsupported as exc:
                logger.warning("Tool calling unavailable (%s); using a fixed search plan.", exc)

        # Fallback / safety net: make sure every suggested query was run at least once.
        source_tool = {"arxiv": "search_arxiv", "semantic_scholar": "search_semantic_scholar"}
        for q in queries:
            if q not in tried:
                for src in self.s.sources:
                    if src in source_tool:
                        run_tool(source_tool[src], {"query": q, "max_results": self.s.results_per_query})

        return {
            "candidates": candidates,
            "queries_tried": tried,
            "pending_queries": [],
            "memory": memory,
            "_summary": f"{len(new_ids)} new candidates ({len(candidates)} total){' via tool calling' if used_tools else ''}",
        }

    # ======================================================================= 3. screen
    def screen(self, state: dict) -> dict:
        plan = state["plan"]
        screened = dict(state.get("screened", {}))
        selected = list(state.get("selected", []))
        unscreened = [pid for pid in state["candidates"] if pid not in screened]

        # Cheap lexical pre-ranking so the LLM only screens the most promising ~30 papers.
        if unscreened:
            query = state["question"] + " " + " ".join(plan.get("key_concepts", [])) + " " + " ".join(s["question"] for s in plan["sub_questions"])
            docs = [self._paper(state, pid) for pid in unscreened]
            scores = BM25().fit([tokenize(p.title + " " + p.abstract) for p in docs]).scores(tokenize(query))
            ranked = [unscreened[i] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)]
            to_screen, skipped = ranked[:30], ranked[30:]
            for pid in skipped:
                screened[pid] = {"paper_id": pid, "relevance": 0, "reason": "not screened (low lexical relevance)", "sub_question_ids": []}
            for batch in chunked(to_screen, 8):
                papers = [self._paper(state, pid).model_dump() for pid in batch]
                try:
                    res = self.llm.structured(screen_messages(state["question"], plan["sub_questions"], plan.get("inclusion_criteria", ""), papers),
                                              ScreeningResult, task="screen", context={"question": state["question"], "papers": papers})
                    got = {d.paper_id: d.model_dump() for d in res.decisions if d.paper_id in batch}
                except Exception as exc:
                    logger.warning("screening batch failed: %s", exc)
                    got = {}
                for pid in batch:
                    screened[pid] = got.get(pid, {"paper_id": pid, "relevance": 3, "reason": "no decision returned", "sub_question_ids": []})

        budget = max(0, min(self.s.papers_per_iteration, self.s.max_papers - len(selected)))
        pool = sorted((d for pid, d in screened.items() if pid not in selected), key=lambda d: d["relevance"], reverse=True)
        picks = [d["paper_id"] for d in pool if d["relevance"] >= MIN_RELEVANCE][:budget]
        if not picks and not selected and pool:  # failure recovery: nothing passed the bar
            picks = [d["paper_id"] for d in pool[:2]]
            logger.warning("No paper reached relevance %d; reading the top 2 anyway.", MIN_RELEVANCE)
        selected += picks
        titles = "; ".join(truncate(state["candidates"][p]["title"], 60) for p in picks)
        return {
            "screened": screened,
            "selected": selected,
            "memory": self._note(state, f"Selected {len(picks)} papers: {titles}"),
            "_summary": f"screened {len(unscreened)}, selected {len(picks)} (total {len(selected)})",
        }

    # ======================================================================= 4. read
    def read(self, state: dict) -> dict:
        read_status = dict(state.get("read_status", {}))
        new = [pid for pid in state.get("selected", []) if pid not in read_status]
        n_chunks = 0
        for pid in new:
            paper = self._paper(state, pid)
            chunks, status = self.reader.read(paper)
            self.index.add(chunks, title=paper.title)
            read_status[pid] = status
            n_chunks += len(chunks)
        degraded = sum(1 for pid in new if read_status[pid].startswith("abstract_only"))
        return {
            "read_status": read_status,
            "memory": self._note(state, f"Read {len(new)} papers ({degraded} abstract-only), {n_chunks} chunks"),
            "_summary": f"read {len(new)} papers, {n_chunks} chunks, {degraded} abstract-only",
        }

    # ======================================================================= 5. extract (RAG)
    def extract(self, state: dict) -> dict:
        self._ensure_indexed(state)
        findings = list(state.get("findings", []))
        seen = set(state.get("seen_passages", []))
        paper_ids = set(state.get("read_status", {}))
        iteration = state.get("iteration", 1)
        added = 0
        for sq in state["plan"]["sub_questions"]:
            hits = self.index.search(sq["question"], k=20, paper_ids=paper_ids)
            hits = self.reranker.rerank(sq["question"], hits, top_n=self.s.passages_per_question)
            hits = [h for h in hits if f"{sq['id']}|{h.chunk.chunk_id}" not in seen]
            if not hits:
                continue
            passages = []
            for h in hits:
                p = state["candidates"].get(h.chunk.paper_id, {})
                ref = Paper(**p).short_ref() if p else h.chunk.paper_id
                passages.append({"chunk_id": h.chunk.chunk_id, "paper_id": h.chunk.paper_id, "ref": ref, "section": h.chunk.section, "text": h.chunk.text})
                seen.add(f"{sq['id']}|{h.chunk.chunk_id}")
            try:
                res = self.llm.structured(extract_messages(state["question"], sq, passages), ExtractionResult, task="extract",
                                          context={"question": state["question"], "sub_question": sq, "passages": passages})
            except Exception as exc:
                logger.warning("extraction failed for %s: %s", sq["id"], exc)
                continue
            by_id = {p["chunk_id"]: p for p in passages}
            for ef in res.findings:
                ef.chunk_id = ef.chunk_id.strip().strip("[]` ")
                if ef.chunk_id not in by_id:  # model cited a passage it was not given -> drop
                    continue
                pid = by_id[ef.chunk_id]["paper_id"]
                if any(f["paper_id"] == pid and f["sub_question_id"] == sq["id"] and token_overlap(ef.claim, f["claim"]) > 0.9 for f in findings):
                    continue  # near-duplicate
                findings.append(Finding(finding_id=f"F{len(findings) + 1}", sub_question_id=sq["id"], claim=ef.claim, paper_id=pid,
                                        chunk_id=ef.chunk_id, evidence=ef.evidence, finding_type=ef.finding_type, iteration=iteration).model_dump())
                added += 1
        return {
            "findings": findings,
            "seen_passages": sorted(seen),
            "memory": self._note(state, f"Extracted {added} new findings"),
            "_summary": f"{added} new findings ({len(findings)} total)",
        }

    # ======================================================================= 6. verify
    def verify(self, state: dict) -> dict:
        findings = self._findings(state)
        todo = [f for f in findings if f.verdict is None]
        to_judge = []
        for f in todo:
            chunk = self.store.get_chunk(f.chunk_id)
            if chunk is None:
                f.quote_match, f.verdict, f.verdict_rationale = False, "unsupported", "cited passage not found"
                continue
            ok, ratio = fuzzy_contains(f.evidence, chunk.text)
            f.quote_match = ok
            if not ok:  # cheap deterministic check first: the quote must really be in the passage
                f.verdict, f.verdict_rationale = "unsupported", f"evidence quote not found in passage (similarity {ratio:.2f})"
            else:
                to_judge.append((f, chunk.text))
        for batch in chunked(to_judge, 6):
            items = [{"finding_id": f.finding_id, "claim": f.claim, "passage": text} for f, text in batch]
            try:
                res = self.llm.structured(verify_messages(items), VerificationResult, task="verify", context={"items": items})
                verdicts = {v.finding_id: v for v in res.verdicts}
            except Exception as exc:
                logger.warning("verification batch failed: %s", exc)
                verdicts = {}
            for f, _ in batch:
                v = verdicts.get(f.finding_id)
                f.verdict = v.verdict if v else "unsupported"
                f.verdict_rationale = v.rationale if v else "no verdict returned"
        counts = defaultdict(int)
        for f in todo:
            counts[f.verdict] += 1
        return {
            "findings": [f.model_dump() for f in findings],
            "memory": self._note(state, f"Verified {len(todo)} findings: {dict(counts)}"),
            "_summary": f"checked {len(todo)}: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        }

    # ======================================================================= 7. reflect (self-critique)
    def coverage(self, state: dict) -> list[dict]:
        verified = [f for f in self._findings(state) if f.is_verified]
        out = []
        for sq in state["plan"]["sub_questions"]:
            fs = [f for f in verified if f.sub_question_id == sq["id"]]
            out.append({"id": sq["id"], "question": sq["question"], "n_findings": len(fs), "n_papers": len({f.paper_id for f in fs})})
        return out

    def reflect(self, state: dict) -> dict:
        iteration = state.get("iteration", 1)
        cov = self.coverage(state)
        verified = [f for f in self._findings(state) if f.is_verified]
        digest = "\n".join(f"- ({f.sub_question_id}) {truncate(f.claim, 160)}" for f in verified[:40]) or "(none)"
        memory = WorkingMemory.compress(state.get("memory") or WorkingMemory.empty(), self.llm)
        try:
            refl = self.llm.structured(
                reflect_messages(state["question"], state["plan"]["sub_questions"], cov, digest, WorkingMemory.render(memory), iteration, self.s.max_iterations),
                Reflection, task="reflect", context={"coverage": cov, "iteration": iteration, "max_iter": self.s.max_iterations})
        except Exception as exc:
            logger.warning("reflection failed (%s); proceeding to synthesis", exc)
            refl = Reflection(sufficient=True, coverage_notes=f"reflection failed: {exc}")

        tried = {q.lower() for q in state.get("queries_tried", [])}
        new_queries = [q for q in refl.new_queries if q.lower() not in tried][:4]
        budget_left = len(state.get("selected", [])) < self.s.max_papers
        loop = (not refl.sufficient) and new_queries and iteration < self.s.max_iterations and budget_left
        memory = WorkingMemory.add(memory, f"Reflection #{iteration}: sufficient={refl.sufficient}; missing={refl.missing_aspects}; next={'search ' + str(new_queries) if loop else 'synthesize'}")
        update = {
            "reflections": list(state.get("reflections", [])) + [{**refl.model_dump(), "iteration": iteration, "coverage": cov}],
            "memory": memory,
            "next_action": "search" if loop else "synthesize",
            "_summary": ("gaps found -> new search: " + "; ".join(new_queries)) if loop else "evidence sufficient -> synthesize",
        }
        if loop:
            update.update({"pending_queries": new_queries, "iteration": iteration + 1})
        return update

    @staticmethod
    def route_after_reflect(state: dict) -> str:
        return state.get("next_action", "synthesize")

    # ======================================================================= 8. synthesize (compare + gaps)
    def _verified_with_refs(self, state: dict) -> list[dict]:
        out = []
        for f in self._findings(state):
            if f.is_verified:
                out.append({**f.model_dump(), "paper_ref": Paper(**state["candidates"][f.paper_id]).short_ref()})
        return out

    def synthesize(self, state: dict) -> dict:
        findings = self._verified_with_refs(state)
        if not findings:
            return {"synthesis": Synthesis().model_dump(), "_summary": "no verified findings to synthesise"}
        cited_papers = sorted({f["paper_id"] for f in findings})
        papers = [state["candidates"][p] for p in cited_papers]
        syn = self.llm.structured(synthesize_messages(state["question"], state["plan"]["sub_questions"], findings, papers), Synthesis,
                                  task="synthesize", context={"findings": findings, "papers": papers})
        valid = {f["finding_id"] for f in findings}
        for group in (syn.agreements, syn.disagreements, syn.gaps):
            for item in group:
                item.finding_ids = [i for i in item.finding_ids if i in valid]   # drop hallucinated ids
        syn.method_comparison = [r for r in syn.method_comparison if r.paper_id in cited_papers]
        return {
            "synthesis": syn.model_dump(),
            "_summary": f"{len(syn.agreements)} agreements, {len(syn.disagreements)} tensions, {len(syn.gaps)} gaps",
        }

    # ======================================================================= 9. write
    def citation_map(self, state: dict) -> dict[str, int]:
        order: list[str] = []
        sq_order = [s["id"] for s in state["plan"]["sub_questions"]]
        for f in sorted(self._verified_with_refs(state), key=lambda f: (sq_order.index(f["sub_question_id"]) if f["sub_question_id"] in sq_order else 99, int(f["finding_id"][1:]))):
            if f["paper_id"] not in order:
                order.append(f["paper_id"])
        return {pid: i + 1 for i, pid in enumerate(order)}

    def write(self, state: dict) -> dict:
        cmap = self.citation_map(state)
        findings = [{**f, "cite": cmap[f["paper_id"]]} for f in self._verified_with_refs(state)]
        attempts = state.get("write_attempts", 0) + 1
        if not findings:
            draft = ("## Executive summary\n\nNo verified evidence could be gathered for this question with the current "
                     "search budget. Try rephrasing the question, enabling more sources, or increasing MAX_ITERATIONS.")
        else:
            draft = self.llm.complete(
                write_messages(state["question"], state["plan"]["sub_questions"], findings, state.get("synthesis", {}), state.get("citation_feedback", "")),
                task="write", context={"findings": findings, "sub_questions": state["plan"]["sub_questions"], "synthesis": state.get("synthesis", {})})
        draft = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", draft.strip())
        return {"report_draft": draft, "citation_map": cmap, "write_attempts": attempts, "_summary": f"draft {attempts}: {len(draft.split())} words"}

    # ======================================================================= 10. check citations
    def check_citations(self, state: dict) -> dict:
        draft = state.get("report_draft", "")
        cmap = state.get("citation_map", {})
        valid = set(cmap.values())
        claims_by_num: dict[int, list[str]] = defaultdict(list)
        for f in self._verified_with_refs(state):
            if f["paper_id"] in cmap:
                claims_by_num[cmap[f["paper_id"]]].append(f["claim"] + " " + f["evidence"])

        invalid, weak, uncited, factual = set(), [], [], 0
        section = ""
        for line in draft.splitlines():
            if line.startswith("#"):
                section = line.lower()
                continue
            if "limitations of this review" in section:
                continue
            for sent in split_sentences(line):
                nums = [n for g in CITE_RE.findall(sent) for n in expand_citation_group(g)]
                invalid.update(n for n in nums if n not in valid)   # checked on every sentence
                bare = CITE_RE.sub("", sent).strip()
                if len(tokenize(bare)) < 5:  # too short to be a factual claim
                    continue
                factual += 1
                if not nums:
                    uncited.append(truncate(bare, 100))
                    continue
                good = [n for n in nums if n in valid]
                if good and max(max((token_overlap(bare, c) for c in claims_by_num[n]), default=0) for n in good) < 0.2:
                    weak.append(truncate(bare, 100))

        uncited_ratio = len(uncited) / factual if factual else 0.0
        problems = []
        if invalid:
            problems.append(f"These citation numbers do not exist: {sorted(invalid)}. Use only {sorted(valid)}.")
        if uncited_ratio > 0.3:
            problems.append(f"{len(uncited)} factual sentences have no citation, e.g.: " + " | ".join(uncited[:4]))
        attempts = state.get("write_attempts", 1)
        check = {"invalid_citations": sorted(invalid), "uncited_sentences": len(uncited), "factual_sentences": factual,
                 "uncited_ratio": round(uncited_ratio, 3), "weakly_supported_sentences": weak[:10], "attempt": attempts}

        if problems and attempts < MAX_WRITE_ATTEMPTS:
            return {"citation_check": check, "citation_feedback": "\n".join(problems), "next_action": "rewrite",
                    "_summary": "citation problems -> rewrite: " + "; ".join(p[:80] for p in problems)}

        final_body = draft
        if invalid:  # last resort repair: strip citations that point nowhere
            final_body = CITE_RE.sub(lambda m: "".join(f"[{n}]" for n in expand_citation_group(m.group(1)) if n in valid), draft)
            check["stripped_invalid"] = True
        report_md = render_report(state, final_body)
        path = self.s.reports_dir / f"{state['run_id']}.md"
        path.write_text(report_md, encoding="utf-8")
        (self.s.reports_dir / f"{state['run_id']}.json").write_text(json.dumps({k: v for k, v in state.items() if k not in ("report_draft",)} | {"report_md": report_md, "citation_check": check}, indent=2, default=str), encoding="utf-8")
        self.store.update_run(state["run_id"], status="done", report_path=str(path))
        return {"citation_check": check, "report_md": report_md, "report_path": str(path), "next_action": "end", "status": "done",
                "_summary": f"report saved ({'with repairs' if problems else 'clean'})"}

    @staticmethod
    def route_after_check(state: dict) -> str:
        return "write" if state.get("next_action") == "rewrite" else "end"
