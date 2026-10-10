"""Command-line interface.

    python -m research_agent ask "How do federated gradient boosting methods trade off privacy and accuracy?"
    python -m research_agent ask "..." --offline          # mock LLM + demo papers, no keys needed
    python -m research_agent runs                          # list past runs
    python -m research_agent resume <run_id>               # continue an interrupted run
    python -m research_agent eval --offline                # run the evaluation benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _settings(args):
    if getattr(args, "offline", False):
        os.environ["OFFLINE"] = "1"
    for flag, env in (("max_iterations", "MAX_ITERATIONS"), ("max_papers", "MAX_PAPERS"), ("sources", "SEARCH_SOURCES")):
        val = getattr(args, flag, None)
        if val is not None:
            os.environ[env] = str(val)
    from .config import Settings

    return Settings()


def _print_summary(state: dict) -> None:
    from .evaluation.metrics import compute_metrics

    print("\n" + "=" * 72)
    print(f"Status: {state.get('status')}   Run: {state.get('run_id')}")
    if state.get("report_path"):
        print(f"Report: {state['report_path']}")
    m = compute_metrics(state)
    keys = ["n_candidates", "n_read", "n_findings", "claim_support_rate", "evidence_quote_match_rate",
            "citation_validity", "uncited_sentence_ratio", "sub_question_coverage", "n_cited_papers"]
    print("Metrics: " + ", ".join(f"{k}={m[k]}" for k in keys if k in m))
    if state.get("errors"):
        print(f"Recovered errors: {len(state['errors'])} (see report JSON)")
    if state.get("usage"):
        print(f"LLM usage: {state['usage']}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_agent", description="Citation-backed literature research agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="research a question")
    a.add_argument("question")
    a.add_argument("--offline", action="store_true", help="mock LLM + bundled demo papers (no network/keys)")
    a.add_argument("--max-iterations", type=int, dest="max_iterations")
    a.add_argument("--max-papers", type=int, dest="max_papers")
    a.add_argument("--sources", help="comma list: arxiv,semantic_scholar")
    a.add_argument("--print", action="store_true", dest="print_report", help="print the report to stdout")

    r = sub.add_parser("resume", help="resume an interrupted run from its last checkpoint")
    r.add_argument("run_id")
    r.add_argument("--offline", action="store_true")

    sub.add_parser("runs", help="list recent runs")

    e = sub.add_parser("eval", help="run the evaluation benchmark")
    e.add_argument("--offline", action="store_true")
    e.add_argument("--benchmark", help="path to a benchmark JSON file")
    e.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge rubric")
    e.add_argument("--limit", type=int, default=0, help="only run the first N questions")

    args = p.parse_args(argv)
    settings = _settings(args)

    if args.cmd == "runs":
        from .memory import PersistentStore

        for run in PersistentStore(settings.db_path).list_runs():
            print(f"{run['run_id']:<60} {run['status']:<10} {run['report_path'] or ''}")
        return 0

    if args.cmd == "eval":
        from .evaluation.run_eval import run_benchmark

        summary = run_benchmark(settings, benchmark_path=args.benchmark, judge=not args.no_judge, limit=args.limit)
        print(json.dumps(summary, indent=2))
        return 0

    from .agent import ResearchAgent

    agent = ResearchAgent(settings)
    state = agent.run(args.question) if args.cmd == "ask" else agent.resume(args.run_id)
    if getattr(args, "print_report", False) and state.get("report_md"):
        print(state["report_md"])
    _print_summary(state)
    return 0 if state.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
