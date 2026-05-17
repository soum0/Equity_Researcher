from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from pipeline.checkpointer import PipelineCheckpointer
from pipeline.display import PipelineDisplay
from pipeline.nodes import (
    node_build_graph,
    node_fundamental,
    node_generate_pdf,
    node_ingest,
    node_lead_analyst,
    node_risk,
    node_sentiment,
)
from pipeline.state import PipelineState

display = PipelineDisplay()


def build_pipeline_graph():
    """
    Build the LangGraph pipeline.

    Graph structure:
    START → ingest → build_graph
          → [fundamental, risk, sentiment]  (parallel fan-out)
          → lead_analyst → pdf → END
    """
    graph = StateGraph(PipelineState)

    graph.add_node("ingest",       node_ingest)
    graph.add_node("build_graph",  node_build_graph)
    graph.add_node("fundamental",  node_fundamental)
    graph.add_node("risk",         node_risk)
    graph.add_node("sentiment",    node_sentiment)
    graph.add_node("lead_analyst", node_lead_analyst)
    graph.add_node("pdf",          node_generate_pdf)

    # Sequential: start → ingest → build_graph
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "build_graph")

    # Fan-out: build_graph → all three agents in parallel
    graph.add_edge("build_graph", "fundamental")
    graph.add_edge("build_graph", "risk")
    graph.add_edge("build_graph", "sentiment")

    # Fan-in: all three → lead_analyst (waits for all)
    graph.add_edge("fundamental", "lead_analyst")
    graph.add_edge("risk",        "lead_analyst")
    graph.add_edge("sentiment",   "lead_analyst")

    # Final: lead_analyst → pdf → END
    graph.add_edge("lead_analyst", "pdf")
    graph.add_edge("pdf", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_pipeline(ticker: str, refresh: bool = False) -> PipelineState:
    """Run the full pipeline for a ticker."""
    app = build_pipeline_graph()
    run_id = str(uuid.uuid4())

    initial_state: PipelineState = {
        "ticker":               ticker,
        "run_id":               run_id,
        "started_at":           datetime.now(timezone.utc).isoformat(),
        "refresh":              refresh,

        "ingestion_complete":   False,
        "ingestion_report":     {},
        "ingested_files":       [],

        "graph_complete":       False,
        "graph_stats":          {},

        "fundamental_complete": False,
        "fundamental_result":   {},
        "fundamental_score":    0.0,
        "fundamental_verdict":  "",

        "risk_complete":        False,
        "risk_result":          {},
        "risk_score":           0.0,
        "risk_verdict":         "",

        "sentiment_complete":   False,
        "sentiment_result":     {},
        "sentiment_score":      0.0,
        "sentiment_verdict":    "",

        "lead_complete":        False,
        "memo_result":          {},
        "recommendation":       "",
        "confidence":           0,
        "price_target":         0.0,
        "composite_score":      0.0,

        "pdf_complete":         False,
        "pdf_path":             "",

        "errors":               [],
        "warnings":             [],
        "node_timings":         {},
    }

    # If resuming, show latest checkpoint
    cp = PipelineCheckpointer(ticker)
    latest = cp.get_latest_checkpoint()
    if latest and not refresh:
        print(f"  Resuming from checkpoint: {latest}")

    # Get company name for header (best effort)
    company_name = ""
    try:
        import json, os
        profile_path = f"data/{ticker}/company_profile.json"
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                company_name = json.load(f).get("name", "")
    except Exception:
        pass

    display.print_pipeline_header(ticker, company_name)

    # Stream pipeline execution
    config = {"configurable": {"thread_id": f"{ticker}_{run_id}"}}
    for event in app.stream(initial_state, config):
        # Node display is handled inside each node function
        pass

    # Get final merged state
    final = app.get_state(config).values

    # Save full state to disk
    cp.save_full_state(dict(final))

    display.print_final_summary(final)
    return final
