"""Wires everything into a runnable agent: dependencies, the state graph, run/resume."""
from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any, Callable, Optional, TypedDict

from .config import Settings
from .graph import END, StateGraph
from .llm import BaseLLM, make_llm
from .memory import PersistentStore, WorkingMemory
from .nodes import Nodes
from .retrieval import Embedder, HybridIndex, Reranker
from .tools import PaperReader, SearchTools
from .utils import logger


class AgentState(TypedDict, total=False):
    """Documentation of every key in the agent state (all JSON-serialisable)."""
    run_id: str
    question: str
    status: str                      # running | done | failed
    plan: dict                       # ResearchPlan
    iteration: int                   # search/read/extract loop counter
    pending_queries: list[str]
    queries_tried: list[str]
    candidates: dict[str, dict]      # paper_id -> Paper
    screened: dict[str, dict]        # paper_id -> ScreeningDecision
    selected: list[str]              # paper ids chosen for reading
    read_status: dict[str, str]      # paper_id -> cached | fulltext | demo | abstract_only:...
    seen_passages: list[str]         # "SQ1|chunk_id" pairs already sent to extraction
    findings: list[dict]             # Finding (with verification fields)
    reflections: list[dict]
    next_action: str                 # router signal
    synthesis: dict
    citation_map: dict[str, int]     # paper_id -> [n]
    report_draft: str
    write_attempts: int
    citation_feedback: str
    citation_check: dict
    report_md: str
    report_path: str
    memory: dict                     # short-term working memory
    errors: list[dict]
    trace: list[dict]


NODE_ORDER = ["plan", "search", "screen", "read", "extract", "verify", "reflect", "synthesize", "write", "check_citations"]


class ResearchAgent:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[BaseLLM] = None,
                 on_event: Optional[Callable[[dict], None]] = None) -> None:
        self.s = settings or Settings()
        self.llm = llm or make_llm(self.s)
        self.store = PersistentStore(self.s.db_path)
        self.tools = SearchTools(self.s, self.store)
        self.reader = PaperReader(self.s, self.store)
        self.index = HybridIndex(Embedder(self.s), self.store)
        self.reranker = Reranker(self.s, self.llm)
        self.nodes = Nodes(self.s, self.llm, self.store, self.tools, self.reader, self.index, self.reranker)
        self.on_event = on_event or _log_event
        self._run_id = ""

    # ------------------------------------------------------------------ graph
    def build_graph(self) -> StateGraph:
        n = self.nodes
        g = StateGraph()
        for name in NODE_ORDER:
            g.add_node(name, getattr(n, name))
        g.set_entry_point("plan")
        g.add_edge("plan", "search")
        g.add_edge("search", "screen")
        g.add_edge("screen", "read")
        g.add_edge("read", "extract")
        g.add_edge("extract", "verify")
        g.add_edge("verify", "reflect")
        g.add_conditional_edges("reflect", n.route_after_reflect, {"search": "search", "synthesize": "synthesize"})
        g.add_edge("synthesize", "write")
        g.add_edge("write", "check_citations")
        g.add_conditional_edges("check_citations", n.route_after_check, {"write": "write", "end": END})
        return g

    def _compile(self):
        def checkpoint(step: int, node: str, next_node: Optional[str], state: dict) -> None:
            self.store.save_checkpoint(state["run_id"], step, node, next_node, state)
            if state.get("status") == "failed":
                self.store.update_run(state["run_id"], status="failed")

        return self.build_graph().compile(
            checkpointer=checkpoint, on_event=self.on_event,
            critical_nodes={"plan", "synthesize", "write"},  # can't produce a report without these
            node_retries=1, max_steps=60,
        )

    # ------------------------------------------------------------------ run / resume
    def run(self, question: str) -> dict:
        question = question.strip()
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:40].strip("-")
        run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slug}"
        state: dict[str, Any] = {
            "run_id": run_id, "question": question, "status": "running", "iteration": 1,
            "memory": WorkingMemory.empty(), "errors": [], "trace": [],
        }
        self.store.create_run(run_id, question, {"offline": self.s.offline, "model": self.s.llm_model, "llm": self.llm.name})
        logger.info("Run %s started", run_id)
        if os.environ.get("RA_USE_LANGGRAPH") == "1":
            return self._run_langgraph(state)
        final = self._compile().invoke(state)
        final["usage"] = dict(self.llm.usage)
        return final

    def resume(self, run_id: str) -> dict:
        ck = self.store.latest_checkpoint(run_id)
        if ck is None:
            raise ValueError(f"no checkpoint for run {run_id}")
        state = ck["state"]
        if not ck["next_node"]:
            logger.info("Run %s already finished.", run_id)
            return state
        logger.info("Resuming %s at node '%s' (step %d)", run_id, ck["next_node"], ck["step"] + 1)
        state["status"] = "running"
        self.store.update_run(run_id, status="running")
        final = self._compile().invoke(state, start_at=ck["next_node"], step=ck["step"] + 1)
        final["usage"] = dict(self.llm.usage)
        return final

    # ------------------------------------------------------------------ optional LangGraph backend
    def _run_langgraph(self, state: dict) -> dict:
        """Experimental: run the same nodes on LangGraph (pip install langgraph; RA_USE_LANGGRAPH=1)."""
        from langgraph.graph import END as LG_END, StateGraph as LGStateGraph

        def wrap(fn):
            def inner(s: dict) -> dict:
                update = dict(fn(s) or {})
                update.pop("_summary", None)
                return update
            return inner

        n = self.nodes
        g = LGStateGraph(AgentState)
        for name in NODE_ORDER:
            g.add_node(name, wrap(getattr(n, name)))
        g.set_entry_point("plan")
        for a, b in zip(NODE_ORDER[:6], NODE_ORDER[1:7]):
            g.add_edge(a, b)
        g.add_conditional_edges("reflect", n.route_after_reflect, {"search": "search", "synthesize": "synthesize"})
        g.add_edge("synthesize", "write")
        g.add_edge("write", "check_citations")
        g.add_conditional_edges("check_citations", n.route_after_check, {"write": "write", "end": LG_END})
        return g.compile().invoke(state, config={"recursion_limit": 60})


def _log_event(event: dict) -> None:
    if event["type"] == "node_start":
        logger.info("▶ %-16s (step %d)", event["node"], event["step"])
    elif event["type"] == "node_end":
        logger.info("✔ %-16s %5.1fs  %s", event["node"], event["seconds"], event.get("summary", ""))
    elif event["type"] == "node_error":
        logger.error("✖ %-16s %s", event["node"], event["error"])
