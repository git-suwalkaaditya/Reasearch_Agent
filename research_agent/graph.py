"""A tiny state-graph engine (the core idea behind LangGraph), written from scratch.

* The agent's **state** is a plain JSON-serialisable dict.
* Each **node** is a function ``state -> partial update`` (a dict merged into the state).
* **Edges** say which node runs next; **conditional edges** call a router function on
  the state to choose the next node - this is how loops (reflect -> search again) work.
* After every node the engine **checkpoints** the state, so a crashed or interrupted
  run can resume from the last completed node.

The API mirrors LangGraph (``add_node``, ``add_edge``, ``add_conditional_edges``,
``set_entry_point``, ``compile().invoke()``) so moving to LangGraph later is easy.
"""
from __future__ import annotations

import time
import traceback
from typing import Callable, Iterator, Optional

from .utils import logger

END = "__end__"
NodeFn = Callable[[dict], dict]
Router = Callable[[dict], str]


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Router, Optional[dict[str, str]]]] = {}
        self.entry: Optional[str] = None

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Router, mapping: Optional[dict[str, str]] = None) -> "StateGraph":
        self.conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def compile(self, **kw) -> "CompiledGraph":
        missing = [n for n in list(self.edges.values()) + list(self.edges) if n != END and n not in self.nodes]
        if missing:
            raise ValueError(f"edges reference unknown nodes: {missing}")
        if not self.entry:
            raise ValueError("entry point not set")
        return CompiledGraph(self, **kw)


class CompiledGraph:
    def __init__(
        self,
        graph: StateGraph,
        checkpointer: Optional[Callable[[int, str, Optional[str], dict], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        critical_nodes: Optional[set[str]] = None,
        node_retries: int = 1,
        max_steps: int = 60,
    ) -> None:
        self.g = graph
        self.checkpointer = checkpointer
        self.on_event = on_event or (lambda e: None)
        self.critical = critical_nodes or set()
        self.node_retries = node_retries
        self.max_steps = max_steps

    def next_node(self, node: str, state: dict) -> str:
        if node in self.g.conditional:
            router, mapping = self.g.conditional[node]
            key = router(state)
            return mapping.get(key, key) if mapping else key
        return self.g.edges.get(node, END)

    def stream(self, state: dict, start_at: Optional[str] = None, step: int = 0) -> Iterator[tuple[str, dict]]:
        node = start_at or self.g.entry
        state = dict(state)
        while node and node != END:
            if step >= self.max_steps:
                logger.error("max_steps (%d) reached - stopping to avoid an infinite loop", self.max_steps)
                state["status"] = "stopped_max_steps"
                break
            self.on_event({"type": "node_start", "node": node, "step": step})
            t0 = time.time()
            update, error = None, None
            for attempt in range(self.node_retries + 1):
                try:
                    update = self.g.nodes[node](state) or {}
                    error = None
                    break
                except Exception as exc:  # node-level failure recovery
                    error = exc
                    logger.warning("node '%s' failed (attempt %d): %s", node, attempt + 1, exc)
            seconds = round(time.time() - t0, 2)

            if error is not None:
                err = {"node": node, "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exception_only(type(error), error)[-1].strip()}
                state["errors"] = list(state.get("errors", [])) + [err]
                self.on_event({"type": "node_error", "node": node, "step": step, "error": err["error"]})
                if node in self.critical:
                    state["status"] = "failed"
                    if self.checkpointer:
                        self.checkpointer(step, node, node, state)  # resume will retry this node
                    yield node, state
                    return
                update = {}

            summary = update.pop("_summary", "") if isinstance(update, dict) else ""
            state.update(update)
            state["trace"] = list(state.get("trace", [])) + [{"step": step, "node": node, "seconds": seconds, "summary": summary, "ok": error is None}]
            nxt = self.next_node(node, state)
            if self.checkpointer:
                self.checkpointer(step, node, None if nxt == END else nxt, state)
            self.on_event({"type": "node_end", "node": node, "step": step, "seconds": seconds, "summary": summary, "next": nxt})
            yield node, state
            node, step = nxt, step + 1

    def invoke(self, state: dict, start_at: Optional[str] = None, step: int = 0) -> dict:
        final = state
        for _, final in self.stream(state, start_at=start_at, step=step):
            pass
        return final
