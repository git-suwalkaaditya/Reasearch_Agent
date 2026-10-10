"""Tests run fully offline (mock LLM + demo papers).  Run with:  python -m pytest -q   (or python -m unittest)"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest


def offline_settings(**env):
    tmp = tempfile.mkdtemp(prefix="ra-test-")
    os.environ.update({"OFFLINE": "1", "DATA_DIR": tmp, **{k: str(v) for k, v in env.items()}})
    from research_agent.config import Settings

    return Settings()


QUESTION = "How do federated gradient boosting methods trade off privacy, accuracy and communication cost?"


class TestRetrieval(unittest.TestCase):
    def test_bm25_ranks_matching_doc_first(self):
        from research_agent.retrieval.bm25 import BM25
        from research_agent.utils import tokenize

        docs = ["federated boosting with differential privacy", "crop disease detection images", "robot path planning"]
        scores = BM25().fit([tokenize(d) for d in docs]).scores(tokenize("differential privacy boosting"))
        self.assertEqual(int(scores.argmax()), 0)

    def test_hybrid_search_and_filter(self):
        from research_agent.memory import PersistentStore
        from research_agent.retrieval import Embedder, HybridIndex
        from research_agent.tools import PaperReader, load_demo_papers

        s = offline_settings()
        store = PersistentStore(s.db_path)
        reader, idx = PaperReader(s, store), HybridIndex(Embedder(s), store)
        papers, _ = load_demo_papers()
        for p in papers:
            store.upsert_paper(p)
            idx.add(reader.read(p)[0], p.title)
        hits = idx.search("membership inference attack accuracy", k=5)
        self.assertEqual(hits[0].chunk.paper_id, "demo:0005")
        only = idx.search("membership inference attack", k=5, paper_ids={"demo:0001"})
        self.assertTrue(all(h.chunk.paper_id == "demo:0001" for h in only))


class TestParsingAndReading(unittest.TestCase):
    def test_parse_arxiv_atom(self):
        from research_agent.tools import parse_arxiv_atom

        xml = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry><id>http://arxiv.org/abs/2101.00001v2</id><published>2021-01-01T00:00:00Z</published>
          <title>A  Test
          Paper</title><summary>Abstract text.</summary><author><name>Jane Doe</name></author>
          <link title="pdf" href="http://arxiv.org/pdf/2101.00001v2"/></entry></feed>"""
        (p,) = parse_arxiv_atom(xml)
        self.assertEqual((p.paper_id, p.title, p.year, p.authors), ("arxiv:2101.00001", "A Test Paper", 2021, ["Jane Doe"]))

    def test_parse_semantic_scholar_prefers_arxiv_id(self):
        from research_agent.tools import parse_semantic_scholar

        data = {"data": [{"paperId": "abc", "title": "T", "year": 2020, "externalIds": {"ArXiv": "2001.1"}, "authors": [{"name": "A"}]},
                         {"paperId": "xyz", "title": "U", "externalIds": {}}]}
        ids = [p.paper_id for p in parse_semantic_scholar(data)]
        self.assertEqual(ids, ["arxiv:2001.1", "s2:xyz"])

    def test_split_sections_drops_references(self):
        from research_agent.tools import split_sections

        text = "1 Introduction\n" + "Intro words here. " * 10 + "\n2. Methods\n" + "Method words here. " * 10 + "\nReferences\n[1] Some ref " * 5
        names = [n for n, _ in split_sections(text)]
        self.assertEqual(names, ["Introduction", "Methods"])

    def test_reader_falls_back_to_abstract(self):
        from research_agent.config import Settings
        from research_agent.memory import PersistentStore
        from research_agent.schemas import Paper
        from research_agent.tools import PaperReader

        offline_settings()
        os.environ["OFFLINE"] = "0"
        s = Settings()
        os.environ["OFFLINE"] = "1"
        reader = PaperReader(s, PersistentStore(s.db_path))
        chunks, status = reader.read(Paper(paper_id="x:1", title="No PDF", abstract="Only an abstract is available for this paper."))
        self.assertTrue(status.startswith("abstract_only"))
        self.assertEqual(len(chunks), 1)


class TestLLMClient(unittest.TestCase):
    def test_structured_output_self_repair(self):
        from research_agent.llm import OpenAICompatLLM
        from research_agent.schemas import Reflection

        class FakeLLM(OpenAICompatLLM):
            replies = ['{"sufficient": "maybe"', '```json\n{"sufficient": true, "new_queries": []}\n```']

            def complete(self, messages, task="", context=None, **kw):
                return self.replies.pop(0)

        os.environ.update({"LLM_MODEL": "x", "LLM_API_KEY": "y"})
        s = offline_settings()
        llm = FakeLLM(s)
        out = llm.structured([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], Reflection, task="t")
        self.assertTrue(out.sufficient)
        self.assertEqual(llm.usage["repairs"], 1)


class TestAgentEndToEnd(unittest.TestCase):
    def test_full_run_produces_verified_cited_report(self):
        from research_agent.agent import ResearchAgent
        from research_agent.evaluation import compute_metrics

        state = ResearchAgent(offline_settings()).run(QUESTION)
        self.assertEqual(state["status"], "done")
        m = compute_metrics(state)
        self.assertGreater(m["n_verified"], 3)
        self.assertEqual(m["citation_validity"], 1.0)
        self.assertEqual(m["sub_question_coverage"], 1.0)
        self.assertIn("## References", state["report_md"])
        self.assertNotIn("demo:0101", state["selected"])  # off-topic paper screened out
        self.assertTrue(os.path.exists(state["report_path"]))

    def test_reflection_loop_runs_second_iteration(self):
        from research_agent.agent import ResearchAgent

        state = ResearchAgent(offline_settings(PAPERS_PER_ITERATION=2, MAX_ITERATIONS=3)).run(QUESTION)
        nodes = [t["node"] for t in state["trace"]]
        self.assertGreaterEqual(nodes.count("search"), 2)
        self.assertGreaterEqual(state["iteration"], 2)
        os.environ.pop("PAPERS_PER_ITERATION"); os.environ.pop("MAX_ITERATIONS")

    def test_citation_check_triggers_rewrite(self):
        from research_agent.agent import ResearchAgent
        from research_agent.mock import MockLLM

        state = ResearchAgent(offline_settings(), llm=MockLLM(bad_first_draft=True)).run(QUESTION)
        self.assertEqual(state["write_attempts"], 2)
        self.assertNotIn("[99]", state["report_md"])

    def test_tool_calling_fallback(self):
        from research_agent.agent import ResearchAgent
        from research_agent.llm import ToolCallingUnsupported
        from research_agent.mock import MockLLM

        class NoTools(MockLLM):
            def chat_with_tools(self, *a, **k):
                raise ToolCallingUnsupported("not supported")

        state = ResearchAgent(offline_settings(), llm=NoTools()).run(QUESTION)
        self.assertEqual(state["status"], "done")
        self.assertGreater(len(state["candidates"]), 3)

    def test_crash_and_resume(self):
        from research_agent.agent import ResearchAgent

        s = offline_settings()
        agent = ResearchAgent(s)
        agent.nodes.synthesize = lambda state: (_ for _ in ()).throw(RuntimeError("simulated outage"))
        crashed = agent.run(QUESTION)
        self.assertEqual(crashed["status"], "failed")
        self.assertEqual(agent.store.latest_checkpoint(crashed["run_id"])["next_node"], "synthesize")

        agent2 = ResearchAgent(s)  # fresh process: empty in-memory index, rebuilt from SQLite
        done = agent2.resume(crashed["run_id"])
        self.assertEqual(done["status"], "done")
        self.assertTrue(done["report_md"])


class TestEvaluation(unittest.TestCase):
    def test_offline_benchmark(self):
        from research_agent.evaluation.run_eval import run_benchmark

        summary = run_benchmark(offline_settings(), judge=True)
        self.assertEqual(summary["n_done"], 2)
        self.assertGreater(summary["recall_candidates"], 0.9)
        self.assertIn("judge_mean", summary)

    def test_citation_metrics_detect_invalid(self):
        from research_agent.evaluation.metrics import citation_metrics

        state = {"citation_map": {"p1": 1}, "findings": [], "report_md": "## Findings\n\nThe method improves accuracy a lot here [1]. It also reduces cost a lot here [7]."}
        m = citation_metrics(state)
        self.assertEqual(m["citation_validity"], 0.5)


if __name__ == "__main__":
    unittest.main()


class TestOpenAICompatibleWire(unittest.TestCase):
    """Exercise the real HTTP client against a local fake OpenAI-compatible server."""

    def test_json_mode_fallback_and_tool_call_parsing(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from research_agent.llm import OpenAICompatLLM
        from research_agent.schemas import Reflection

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body)
                if "response_format" in body:
                    self.send_response(400); self.end_headers()
                    self.wfile.write(b'{"error": "response_format is not supported"}')
                    return
                if "tools" in body:
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "search_arxiv", "arguments": '{"query": "dp boosting"}'}}]}
                else:
                    msg = {"role": "assistant", "content": '{"sufficient": false, "new_queries": ["a"]}'}
                out = json.dumps({"choices": [{"message": msg}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(out)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            os.environ.update({"LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "LLM_MODEL": "fake", "LLM_API_KEY": "k"})
            s = offline_settings()
            llm = OpenAICompatLLM(s)
            msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            r = llm.structured(msgs, Reflection, task="reflect")
            self.assertFalse(r.sufficient)
            self.assertFalse(llm._json_mode)              # degraded gracefully after the 400
            res = llm.chat_with_tools(msgs, [{"type": "function", "function": {"name": "search_arxiv", "parameters": {}}}])
            self.assertEqual(res["tool_calls"][0]["arguments"], {"query": "dp boosting"})
            self.assertEqual(llm.usage["prompt_tokens"], 10)
        finally:
            server.shutdown()
            os.environ.pop("LLM_BASE_URL", None)
