import json
import os
import time

import streamlit as st


def render():
    st.markdown("## 🚀 Run Pipeline Analysis")

    ticker = st.session_state.get("selected_ticker", "")
    ticker_input = st.text_input(
        "Ticker Symbol", value=ticker, placeholder="AAPL"
    ).upper().strip()

    col1, col2, col3 = st.columns(3)
    with col1:
        refresh = st.checkbox("Force refresh (ignore cache)", value=False)
    with col2:
        skip_graph = st.checkbox(
            "Skip graph rebuild (use existing)", value=True,
            help="Much faster if graph already built",
        )
    with col3:
        st.markdown("")

    run_btn = st.button(
        "🚀 Start Analysis", type="primary",
        use_container_width=True,
        disabled=not bool(ticker_input),
    )

    if not run_btn:
        if ticker_input and os.path.exists(
            f"data/{ticker_input}/analysis/investment_memo.json"
        ):
            st.success(f"✅ Analysis exists for {ticker_input}. View it in Investment Memo tab.")
            if st.button("📋 View Investment Memo"):
                st.session_state["selected_ticker"] = ticker_input
                st.session_state["current_page"] = "📋 Investment Memo"
                st.rerun()
        return

    # ── Pipeline execution ─────────────────────────────────────────────────
    st.session_state["selected_ticker"] = ticker_input
    st.markdown("---")

    header_ph   = st.empty()
    progress_ph = st.empty()
    status_ph   = st.empty()
    output_ph   = st.empty()

    header_ph.markdown(f"### 🔄 Analyzing **{ticker_input}**...")

    NODES = [
        ("ingest",      "📥 Data Ingestion",      25),
        ("build_graph", "🕸️  GraphRAG Build",      780 if not skip_graph else 5),
        ("fundamental", "🔍 Fundamental Analysis", 100),
        ("risk",        "⚠️  Risk Analysis",        100),
        ("sentiment",   "📰 Sentiment Analysis",   210),
        ("lead",        "🎯 Lead Analyst",          20),
        ("pdf",         "📄 PDF Generation",        5),
    ]

    results_log  = []
    progress_bar = progress_ph.progress(0)
    total_weight = sum(w for _, _, w in NODES)
    cumulative_weight = 0

    def run_node(node_name: str, display_name: str) -> dict:
        status_ph.info(f"🔄 Running: {display_name}...")
        start = time.time()
        try:
            from pipeline.nodes import (
                node_build_graph,
                node_fundamental,
                node_generate_pdf,
                node_ingest,
                node_lead_analyst,
                node_risk,
                node_sentiment,
            )

            # Build state, seeding from any saved state
            state: dict = {
                "ticker": ticker_input, "refresh": refresh,
                "ingestion_complete": False, "graph_complete": False,
                "fundamental_complete": False, "risk_complete": False,
                "sentiment_complete": False, "lead_complete": False,
                "pdf_complete": False,
                "errors": [], "warnings": [], "node_timings": {},
            }
            state_file = f"data/{ticker_input}/pipeline/state.json"
            if os.path.exists(state_file):
                with open(state_file) as f:
                    state.update(json.load(f))
            # Ensure mutable defaults not shared
            state.setdefault("errors", [])
            state.setdefault("warnings", [])
            state.setdefault("node_timings", {})

            node_fns = {
                "ingest":      node_ingest,
                "build_graph": node_build_graph,
                "fundamental": node_fundamental,
                "risk":        node_risk,
                "sentiment":   node_sentiment,
                "lead":        node_lead_analyst,
                "pdf":         node_generate_pdf,
            }

            # Honour skip_graph
            if node_name == "build_graph" and skip_graph:
                report_path = f"data/{ticker_input}/graphrag/graphrag_report.json"
                if os.path.exists(report_path):
                    with open(report_path) as f:
                        stats = json.load(f)
                    state["graph_complete"] = True
                    state["graph_stats"] = stats
                    _save_state(ticker_input, state)
                    return {
                        "status": "skipped",
                        "message": "Graph loaded from existing index",
                        "elapsed": time.time() - start,
                    }

            result = node_fns[node_name](state)
            # Merge updates back into state and persist
            state.update(result)
            _save_state(ticker_input, state)
            return {"status": "ok", "result": result, "elapsed": time.time() - start}

        except Exception as e:
            return {"status": "error", "error": str(e), "elapsed": time.time() - start}

    for node_name, display_name, weight in NODES:
        node_result = run_node(node_name, display_name)
        cumulative_weight += weight
        progress_bar.progress(min(cumulative_weight / total_weight, 1.0))

        elapsed = node_result.get("elapsed", 0)
        if node_result["status"] == "skipped":
            status_str = f"skipped (cached) in {elapsed:.1f}s"
            icon = "⏭️"
        elif node_result["status"] == "ok":
            status_str = f"completed in {elapsed:.1f}s"
            icon = "✅"
        else:
            status_str = f"ERROR: {node_result.get('error', 'unknown')}"
            icon = "❌"

        results_log.append(f"{icon} {display_name} — {status_str}")
        output_ph.markdown("\n".join(results_log))

        if node_result["status"] == "error":
            status_ph.error(f"❌ Pipeline failed at {display_name}: {node_result.get('error')}")
            return

    progress_bar.progress(1.0)
    status_ph.success("✅ Pipeline complete!")
    header_ph.markdown(f"### ✅ Analysis Complete — **{ticker_input}**")
    time.sleep(1)
    st.session_state["current_page"] = "📋 Investment Memo"
    st.rerun()


def _save_state(ticker: str, state: dict):
    """Persist merged state so subsequent nodes see prior results."""
    out_dir = f"data/{ticker}/pipeline"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/state.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
