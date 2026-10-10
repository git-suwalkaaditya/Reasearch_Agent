"""Streamlit UI for the research agent.

    pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Research Agent", page_icon="🔬", layout="wide")

DEFAULT_Q = "How do federated gradient boosting methods trade off privacy, accuracy and communication cost?"

# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    offline = st.toggle("Offline demo mode", value=not bool(os.environ.get("LLM_API_KEY")),
                        help="Mock LLM + bundled fictional demo papers. No API key or internet needed.")
    max_iter = st.slider("Max search iterations", 1, 4, int(os.environ.get("MAX_ITERATIONS", 2)))
    max_papers = st.slider("Max papers to read", 2, 20, int(os.environ.get("MAX_PAPERS", 8)))
    sources = st.multiselect("Search sources", ["arxiv", "semantic_scholar"], default=["arxiv", "semantic_scholar"])
    if not offline:
        st.caption(f"LLM: `{os.environ.get('LLM_MODEL', 'not set')}` @ `{os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')}`")

os.environ.update({
    "OFFLINE": "1" if offline else "0",
    "MAX_ITERATIONS": str(max_iter),
    "MAX_PAPERS": str(max_papers),
    "SEARCH_SOURCES": ",".join(sources) or "arxiv",
})

from research_agent.agent import ResearchAgent  # noqa: E402  (import after env is set)
from research_agent.config import Settings  # noqa: E402
from research_agent.evaluation import compute_metrics  # noqa: E402
from research_agent.llm import LLMError  # noqa: E402
from research_agent.memory import PersistentStore  # noqa: E402

settings = Settings()

with st.sidebar:
    st.header("🗂️ Past runs")
    runs = PersistentStore(settings.db_path).list_runs(limit=15)
    if runs:
        labels = {f"{r['status']} · {r['question'][:50]}": r["run_id"] for r in runs}
        choice = st.selectbox("Open a previous run", ["—"] + list(labels))
        c1, c2 = st.columns(2)
        if choice != "—" and c1.button("Open"):
            ck = PersistentStore(settings.db_path).latest_checkpoint(labels[choice])
            if ck:
                st.session_state["state"] = ck["state"]
        if choice != "—" and c2.button("Resume"):
            st.session_state["resume"] = labels[choice]
    else:
        st.caption("No runs yet.")

# ---------------------------------------------------------------------------- main
st.title("🔬 Research Agent")
st.caption("Plans → searches papers → reads sections → verifies claims → compares findings → finds gaps → writes a citation-backed report.")
question = st.text_area("Research question", value=DEFAULT_Q, height=80)
run_clicked = st.button("Run research", type="primary", disabled=not question.strip())

resume_id = st.session_state.pop("resume", None)
if run_clicked or resume_id:
    status = st.status("Researching…", expanded=True)

    def on_event(e: dict) -> None:
        if e["type"] == "node_start":
            status.update(label=f"Running **{e['node']}** (step {e['step']})…")
        elif e["type"] == "node_end":
            status.write(f"✅ **{e['node']}** · {e['seconds']}s · {e.get('summary', '')}")
        elif e["type"] == "node_error":
            status.write(f"⚠️ **{e['node']}** failed: {e['error']}")

    try:
        agent = ResearchAgent(settings, on_event=on_event)
        state = agent.resume(resume_id) if resume_id else agent.run(question)
    except LLMError as exc:
        status.update(label="Configuration error", state="error")
        st.error(f"{exc}\n\nSet LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in `.env`, or switch on offline demo mode.")
        st.stop()
    status.update(label=f"Finished: {state.get('status')}", state="complete" if state.get("status") == "done" else "error", expanded=False)
    st.session_state["state"] = state

state = st.session_state.get("state")
if not state:
    st.info("Enter a question and press **Run research**. Offline demo mode works without any API key.")
    st.stop()

m = compute_metrics(state)
cols = st.columns(6)
cols[0].metric("Papers found", m["n_candidates"])
cols[1].metric("Papers read", m["n_read"])
cols[2].metric("Verified claims", f"{m['n_verified']}/{m['n_findings']}")
cols[3].metric("Citation validity", f"{m['citation_validity']:.0%}")
cols[4].metric("Sub-question coverage", f"{m['sub_question_coverage']:.0%}")
cols[5].metric("Search iterations", m["iterations"])

tab_report, tab_evidence, tab_papers, tab_gaps, tab_trace = st.tabs(["📄 Report", "🔎 Evidence", "📚 Papers", "🧩 Gaps", "🧠 Trace & memory"])

with tab_report:
    if state.get("report_md"):
        st.download_button("Download report (.md)", state["report_md"], file_name=f"{state['run_id']}.md")
        st.markdown(state["report_md"], unsafe_allow_html=True)
    else:
        st.warning("No report yet (the run may have failed - see Trace).")

with tab_evidence:
    cands = state.get("candidates", {})
    rows = [{
        "id": f["finding_id"], "sub_q": f["sub_question_id"], "verdict": f.get("verdict"), "quote found": f.get("quote_match"),
        "claim": f["claim"], "evidence": f["evidence"], "paper": cands.get(f["paper_id"], {}).get("title", f["paper_id"]),
        "passage": f["chunk_id"],
    } for f in state.get("findings", [])]
    if rows:
        df = pd.DataFrame(rows)
        only_verified = st.checkbox("Only verified claims", value=False)
        if only_verified:
            df = df[df["verdict"].isin(["supported", "partially_supported"]) & df["quote found"].fillna(False)]
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_papers:
    screened, read = state.get("screened", {}), state.get("read_status", {})
    rows = [{
        "title": p["title"], "year": p.get("year"), "source": p.get("source"),
        "relevance": screened.get(pid, {}).get("relevance"), "reason": screened.get(pid, {}).get("reason", ""),
        "selected": pid in state.get("selected", []), "read": read.get(pid, ""), "url": p.get("url", ""),
    } for pid, p in state.get("candidates", {}).items()]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("relevance", ascending=False, na_position="last"), use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("link")})

with tab_gaps:
    syn = state.get("synthesis") or {}
    for g in syn.get("gaps", []):
        with st.container(border=True):
            st.markdown(f"**{g['gap']}**")
            st.write(g.get("why_it_matters", ""))
            st.caption(f"Suggested direction: {g.get('suggested_direction', '')} · evidence: {', '.join(g.get('finding_ids', []))}")
    if syn.get("disagreements"):
        st.subheader("Tensions between studies")
        for d in syn["disagreements"]:
            st.markdown(f"- {d['statement']} ({', '.join(d['finding_ids'])})")

with tab_trace:
    st.subheader("Node trace")
    st.dataframe(pd.DataFrame(state.get("trace", [])), use_container_width=True, hide_index=True)
    st.subheader("Plan")
    st.json(state.get("plan", {}), expanded=False)
    st.subheader("Reflections")
    st.json(state.get("reflections", []), expanded=False)
    st.subheader("Working memory")
    mem = state.get("memory", {})
    if mem.get("summary"):
        st.info(mem["summary"])
    st.code("\n".join(mem.get("notes", [])) or "(empty)")
    st.subheader("Citation check")
    st.json(state.get("citation_check", {}), expanded=False)
    if state.get("errors"):
        st.subheader("Recovered errors")
        st.json(state["errors"])
    st.subheader("All metrics")
    st.json(m, expanded=False)
