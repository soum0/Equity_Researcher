from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    """
    Full pipeline state passed between LangGraph nodes.
    Every node reads from and writes to this state.
    Checkpointed to disk after each node completes.
    """
    # ── Core ──────────────────────────────────────────────────
    ticker: str
    run_id: str           # UUID for this pipeline run
    started_at: str       # ISO timestamp
    refresh: bool         # Force re-run all steps

    # ── Step 1: Ingestion ─────────────────────────────────────
    ingestion_complete: bool
    ingestion_report: dict
    ingested_files: list  # [{type, file, records/chars, ...}]

    # ── Step 2: GraphRAG ──────────────────────────────────────
    graph_complete: bool
    graph_stats: dict

    # ── Step 3: Fundamental Agent ─────────────────────────────
    fundamental_complete: bool
    fundamental_result: dict
    fundamental_score: float
    fundamental_verdict: str

    # ── Step 4: Risk Agent ────────────────────────────────────
    risk_complete: bool
    risk_result: dict
    risk_score: float
    risk_verdict: str

    # ── Step 5: Sentiment Agent ───────────────────────────────
    sentiment_complete: bool
    sentiment_result: dict
    sentiment_score: float
    sentiment_verdict: str

    # ── Step 6: Lead Analyst ──────────────────────────────────
    lead_complete: bool
    memo_result: dict
    recommendation: str
    confidence: int
    price_target: float
    composite_score: float

    # ── Step 7: PDF ───────────────────────────────────────────
    pdf_complete: bool
    pdf_path: str

    # ── Merge-able fields (reducers for parallel nodes) ───────
    errors:       Annotated[list, operator.add]
    warnings:     Annotated[list, operator.add]
    node_timings: Annotated[dict, lambda a, b: {**a, **b}]
