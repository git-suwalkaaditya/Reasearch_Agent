"""Memory for the agent.

* ``WorkingMemory`` - short-term memory: a rolling scratchpad of observations made
  during one run (what was searched, what was found, what failed). When it grows too
  long, older notes are compressed into a summary by the LLM so prompts stay small.

* ``PersistentStore`` - long-term memory in SQLite: every paper and chunk ever read,
  cached embeddings, run metadata and a checkpoint of the agent state after every
  node. This gives you (1) a personal paper library that later runs can search,
  (2) no re-downloading/re-embedding, and (3) crash recovery: ``--resume RUN_ID``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .schemas import Chunk, MemorySummary, Paper
from .utils import logger, normalize_title, truncate


# =========================================================================== short-term
class WorkingMemory:
    MAX_NOTES = 14
    KEEP_RECENT = 6

    @staticmethod
    def empty() -> dict:
        return {"summary": "", "notes": []}

    @staticmethod
    def add(memory: dict, note: str) -> dict:
        mem = {"summary": memory.get("summary", ""), "notes": list(memory.get("notes", []))}
        mem["notes"].append(truncate(note, 400))
        return mem

    @staticmethod
    def render(memory: dict, max_chars: int = 2500) -> str:
        parts = []
        if memory.get("summary"):
            parts.append("Summary of earlier work: " + memory["summary"])
        parts += [f"- {n}" for n in memory.get("notes", [])]
        text = "\n".join(parts) or "(nothing yet)"
        return text[-max_chars:]

    @classmethod
    def compress(cls, memory: dict, llm) -> dict:
        """Summarise old notes when the scratchpad gets long (memory consolidation)."""
        notes = memory.get("notes", [])
        if len(notes) <= cls.MAX_NOTES:
            return memory
        old, recent = notes[: -cls.KEEP_RECENT], notes[-cls.KEEP_RECENT :]
        from .prompts import memory_summary_messages

        try:
            res = llm.structured(
                memory_summary_messages(memory.get("summary", ""), old),
                MemorySummary,
                task="memory_summary",
                context={"summary": memory.get("summary", ""), "notes": old},
            )
            summary = res.summary
        except Exception as exc:  # never let memory management kill a run
            logger.warning("memory compression failed: %s", exc)
            summary = (memory.get("summary", "") + " " + " ".join(old))[-1500:]
        return {"summary": truncate(summary, 1500), "notes": recent}


# =========================================================================== long-term
_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY, title_norm TEXT, data TEXT NOT NULL, added REAL);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title_norm);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, section TEXT, position INTEGER, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (chunk_id, model));
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, question TEXT, status TEXT, created REAL, updated REAL, report_path TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER, node TEXT, next_node TEXT, state TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id);
"""


class PersistentStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # -- papers --------------------------------------------------------------
    def upsert_paper(self, paper: Paper) -> None:
        existing = self.get_paper(paper.paper_id)
        if existing:  # keep the richer record (e.g. don't lose a pdf_url)
            merged = existing.model_dump()
            for k, v in paper.model_dump().items():
                if v not in (None, "", [], 0):
                    merged[k] = v
            paper = Paper(**merged)
        self._exec(
            "INSERT OR REPLACE INTO papers (paper_id, title_norm, data, added) VALUES (?,?,?,?)",
            (paper.paper_id, normalize_title(paper.title), paper.model_dump_json(), time.time()),
        )

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        rows = self._query("SELECT data FROM papers WHERE paper_id=?", (paper_id,))
        return Paper.model_validate_json(rows[0]["data"]) if rows else None

    def find_by_title(self, title: str) -> Optional[Paper]:
        rows = self._query("SELECT data FROM papers WHERE title_norm=?", (normalize_title(title),))
        return Paper.model_validate_json(rows[0]["data"]) if rows else None

    def all_papers(self) -> list[Paper]:
        return [Paper.model_validate_json(r["data"]) for r in self._query("SELECT data FROM papers")]

    # -- chunks ---------------------------------------------------------------
    def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            self.conn.execute("DELETE FROM chunks WHERE paper_id=?", (chunks[0].paper_id,))
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, section, position, text) VALUES (?,?,?,?,?)",
                [(c.chunk_id, c.paper_id, c.section, c.position, c.text) for c in chunks],
            )
            self.conn.commit()

    def get_chunks(self, paper_id: str) -> list[Chunk]:
        rows = self._query("SELECT * FROM chunks WHERE paper_id=? ORDER BY position", (paper_id,))
        return [Chunk(chunk_id=r["chunk_id"], paper_id=r["paper_id"], section=r["section"], position=r["position"], text=r["text"]) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        rows = self._query("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        if not rows:
            return None
        r = rows[0]
        return Chunk(chunk_id=r["chunk_id"], paper_id=r["paper_id"], section=r["section"], position=r["position"], text=r["text"])

    def has_fulltext(self, paper_id: str) -> bool:
        rows = self._query("SELECT COUNT(*) AS n FROM chunks WHERE paper_id=? AND section != 'Abstract'", (paper_id,))
        return rows[0]["n"] > 0

    # -- embedding cache --------------------------------------------------------
    def get_embeddings(self, chunk_ids: list[str], model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for r in self._query(f"SELECT chunk_id, vector FROM embeddings WHERE model=? AND chunk_id IN ({marks})", (model, *batch)):
                out[r["chunk_id"]] = np.frombuffer(r["vector"], dtype=np.float32)
        return out

    def put_embeddings(self, vectors: dict[str, np.ndarray], model: str) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (chunk_id, model, vector) VALUES (?,?,?)",
                [(cid, model, np.asarray(v, dtype=np.float32).tobytes()) for cid, v in vectors.items()],
            )
            self.conn.commit()

    # -- runs & checkpoints -----------------------------------------------------
    def create_run(self, run_id: str, question: str, meta: dict) -> None:
        now = time.time()
        self._exec(
            "INSERT OR REPLACE INTO runs (run_id, question, status, created, updated, report_path, meta) VALUES (?,?,?,?,?,?,?)",
            (run_id, question, "running", now, now, "", json.dumps(meta)),
        )

    def update_run(self, run_id: str, status: Optional[str] = None, report_path: Optional[str] = None) -> None:
        if status is not None:
            self._exec("UPDATE runs SET status=?, updated=? WHERE run_id=?", (status, time.time(), run_id))
        if report_path is not None:
            self._exec("UPDATE runs SET report_path=?, updated=? WHERE run_id=?", (report_path, time.time(), run_id))

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._query("SELECT run_id, question, status, created, report_path FROM runs ORDER BY created DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def save_checkpoint(self, run_id: str, step: int, node: str, next_node: Optional[str], state: dict) -> None:
        self._exec(
            "INSERT INTO checkpoints (run_id, step, node, next_node, state, ts) VALUES (?,?,?,?,?,?)",
            (run_id, step, node, next_node or "", json.dumps(state, default=str), time.time()),
        )

    def latest_checkpoint(self, run_id: str) -> Optional[dict[str, Any]]:
        rows = self._query("SELECT * FROM checkpoints WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,))
        if not rows:
            return None
        r = rows[0]
        return {"step": r["step"], "node": r["node"], "next_node": r["next_node"] or None, "state": json.loads(r["state"])}
