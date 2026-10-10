"""Benchmark runner: run the agent on a set of questions and aggregate metrics.

    python -m research_agent eval --offline          # quick, deterministic
    python -m research_agent eval --limit 2          # real LLM + live search

Each benchmark item: {"question": str, "gold_titles": [titles a good review should find]}.
Results are written to <DATA_DIR>/eval/<timestamp>.json and .md.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from statistics import mean
from typing import Optional

from ..config import Settings
from ..utils import logger
from .judge import judge_report
from .metrics import compute_metrics, retrieval_recall

DATA = Path(__file__).resolve().parent.parent / "data"
SUMMARY_KEYS = [
    "claim_support_rate", "evidence_quote_match_rate", "citation_validity", "citation_support_rate",
    "uncited_sentence_ratio", "sub_question_coverage", "recall_candidates", "recall_selected", "judge_mean", "runtime_s",
]


def run_benchmark(settings: Settings, benchmark_path: Optional[str] = None, judge: bool = True, limit: int = 0) -> dict:
    from ..agent import ResearchAgent

    path = Path(benchmark_path) if benchmark_path else DATA / ("benchmark_offline.json" if settings.offline else "benchmark.json")
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    if limit:
        items = items[:limit]
    rows = []
    for i, item in enumerate(items, 1):
        logger.info("=== Benchmark %d/%d: %s", i, len(items), item["question"])
        agent = ResearchAgent(settings)
        try:
            state = agent.run(item["question"])
        except Exception as exc:  # one failing question must not kill the benchmark
            logger.error("run failed: %s", exc)
            rows.append({"question": item["question"], "status": "crashed", "error": str(exc)})
            continue
        m = compute_metrics(state)
        m.update(retrieval_recall(state, item.get("gold_titles", [])))
        if judge and state.get("report_md"):
            try:
                m.update(judge_report(agent.llm, state, m))
            except Exception as exc:
                logger.warning("judge failed: %s", exc)
        rows.append({"question": item["question"], "run_id": state.get("run_id"), **m})

    summary = {k: round(mean(r[k] for r in rows if isinstance(r.get(k), (int, float))), 3)
               for k in SUMMARY_KEYS if any(isinstance(r.get(k), (int, float)) for r in rows)}
    summary["n_questions"] = len(rows)
    summary["n_done"] = sum(1 for r in rows if r.get("status") == "done")

    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    (out_dir / f"{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")
    cols = ["question"] + [k for k in SUMMARY_KEYS if any(k in r for r in rows)]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, ""))[:60] for c in cols) + " |")
    md.append("| **mean** | " + " | ".join(str(summary.get(c, "")) for c in cols[1:]) + " |")
    (out_dir / f"{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Evaluation written to %s", out_dir / f"{stamp}.md")
    return summary
