from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone

import networkx as nx

from config import Config
from pipeline.checkpointer import PipelineCheckpointer
from pipeline.display import PipelineDisplay
from pipeline.state import PipelineState

display = PipelineDisplay()


def _ticker_config(state: PipelineState) -> tuple[str, Config]:
    return state["ticker"], Config()


def node_ingest(state: PipelineState) -> dict:
    """NODE 1: Data Ingestion."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    display.print_node_start("ingest")
    t0 = time.time()

    if cp.should_skip_node("01_ingest", state.get("refresh", False)):
        saved = cp.load("01_ingest")
        display.print_node_complete("ingest", time.time() - t0)
        return {
            "ingestion_complete": saved.get("ingestion_complete", True),
            "ingestion_report":   saved.get("ingestion_report", {}),
            "ingested_files":     saved.get("ingested_files", []),
            "node_timings":       {"ingest": time.time() - t0},
        }

    try:
        from ingestion.pipeline import DataIngestionPipeline
        pipeline = DataIngestionPipeline(ticker, config)
        report = pipeline.run()
    except Exception as e:
        report = {}
        print(f"  [WARN] Ingestion failed: {e}")

    # Load saved report from disk if run() doesn't return it fully
    report_path = f"{config.DATA_DIR}/{ticker}/ingestion_report.json"
    if not report and os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

    ingested_files = _build_ingested_files(ticker, config, report)
    display.print_ingestion_results(ingested_files)

    elapsed = time.time() - t0
    updates = {
        "ingestion_complete": True,
        "ingestion_report":   report,
        "ingested_files":     ingested_files,
        "node_timings":       {"ingest": elapsed},
    }
    cp.save("01_ingest", {**state, **updates})
    display.print_node_complete("ingest", elapsed)
    return updates


def _build_ingested_files(ticker: str, config: Config, report: dict) -> list[dict]:
    summary = report.get("summary", {})
    files = []

    # Price
    price_path = f"{config.DATA_DIR}/{ticker}/price_data.json"
    if os.path.exists(price_path):
        files.append({
            "type": "price",
            "file": "price_data.json",
            "records": summary.get("price_history_days", 0),
        })

    # Profile
    profile_path = f"{config.DATA_DIR}/{ticker}/company_profile.json"
    if os.path.exists(profile_path):
        files.append({
            "type": "profile",
            "file": "company_profile.json",
            "company": summary.get("company_name", ticker),
        })

    # News
    news_path = f"{config.DATA_DIR}/{ticker}/news_articles.json"
    if os.path.exists(news_path):
        files.append({
            "type": "news",
            "file": "news_articles.json",
            "records": summary.get("news_articles_fetched", 0),
        })

    # SEC filings
    meta_path = f"{config.DATA_DIR}/{ticker}/sec_filings/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for filing in meta.get("filings", []):
            files.append({
                "type": f"sec {filing.get('form_type', '')}",
                "file": filing.get("filename", ""),
                "chars": filing.get("char_count", 0),
                "sections": filing.get("sections_detected", []),
            })
    elif summary.get("sec_filings_fetched"):
        files.append({
            "type": "sec",
            "file": "sec_filings/",
            "records": summary.get("sec_filings_fetched", 0),
        })

    # Macro
    macro_path = f"{config.DATA_DIR}/{ticker}/macro_indicators.json"
    if os.path.exists(macro_path):
        files.append({
            "type": "macro",
            "file": "macro_indicators.json",
            "records": summary.get("macro_indicators_fetched", 0),
        })

    return files


def node_build_graph(state: PipelineState) -> dict:
    """NODE 2: GraphRAG Build."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    display.print_node_start("build_graph")
    t0 = time.time()

    if cp.should_skip_node("02_build_graph", state.get("refresh", False)):
        saved = cp.load("02_build_graph")
        display.print_node_complete("build_graph", time.time() - t0)
        return {
            "graph_complete": saved.get("graph_complete", True),
            "graph_stats":    saved.get("graph_stats", {}),
            "node_timings":   {"build_graph": time.time() - t0},
        }

    try:
        from graphrag.pipeline import GraphRAGPipeline
        GraphRAGPipeline(ticker, config).build()
    except Exception as e:
        print(f"  [WARN] Graph build failed: {e}")

    graph_stats = _load_graph_stats(ticker, config)
    display.print_graph_results(graph_stats)

    elapsed = time.time() - t0
    updates = {
        "graph_complete": True,
        "graph_stats":    graph_stats,
        "node_timings":   {"build_graph": elapsed},
    }
    cp.save("02_build_graph", {**state, **updates})
    display.print_node_complete("build_graph", elapsed)
    return updates


def _load_graph_stats(ticker: str, config: Config) -> dict:
    graphrag_dir = f"{config.DATA_DIR}/{ticker}/graphrag"
    report_path = f"{graphrag_dir}/graphrag_report.json"

    stats = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            stats = json.load(f)

    # Enrich with node_type_breakdown and top_connected_nodes from graph.pkl
    graph_path = f"{graphrag_dir}/graph.pkl"
    if os.path.exists(graph_path):
        try:
            with open(graph_path, "rb") as f:
                G = pickle.load(f)

            # Node type breakdown
            type_counts: dict[str, int] = {}
            for _, data in G.nodes(data=True):
                ntype = data.get("type", "Unknown")
                type_counts[ntype] = type_counts.get(ntype, 0) + 1
            stats["node_type_breakdown"] = dict(
                sorted(type_counts.items(), key=lambda x: -x[1])
            )

            # Top connected nodes by degree
            top_degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:3]
            top_nodes = []
            for node_id, degree in top_degrees:
                node_data = G.nodes.get(node_id, {})
                top_nodes.append({
                    "label":  node_data.get("label", node_id),
                    "type":   node_data.get("type", ""),
                    "degree": degree,
                })
            stats["top_connected_nodes"] = top_nodes
        except Exception as e:
            print(f"  [WARN] Could not read graph.pkl: {e}")

    return stats


def node_fundamental(state: PipelineState) -> dict:
    """NODE 3: Fundamental Analysis (runs in parallel with 4 & 5)."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    print(f"\n🔍 Fundamental Agent starting...")
    t0 = time.time()

    if cp.should_skip_node("03_fundamental", state.get("refresh", False)):
        saved = cp.load("03_fundamental")
        elapsed = time.time() - t0
        display.print_node_complete("fundamental", elapsed)
        return {
            "fundamental_complete": saved.get("fundamental_complete", True),
            "fundamental_result":   saved.get("fundamental_result", {}),
            "fundamental_score":    saved.get("fundamental_score", 0.0),
            "fundamental_verdict":  saved.get("fundamental_verdict", ""),
            "node_timings":         {"fundamental": elapsed},
        }

    try:
        from agents.fundamental_agent import FundamentalAgent
        result = FundamentalAgent(ticker, config).analyze(refresh=state.get("refresh", False))
    except Exception as e:
        print(f"  [ERROR] Fundamental agent failed: {e}")
        result = {}
        return {
            "fundamental_complete": False,
            "fundamental_result":   {},
            "fundamental_score":    5.0,
            "fundamental_verdict":  "ERROR",
            "errors":               [{"node": "fundamental", "error": str(e)}],
            "node_timings":         {"fundamental": time.time() - t0},
        }

    display.print_agent_result("fundamental", result)
    elapsed = time.time() - t0
    updates = {
        "fundamental_complete": True,
        "fundamental_result":   result,
        "fundamental_score":    float(result.get("overall_score", 0)),
        "fundamental_verdict":  result.get("verdict", ""),
        "node_timings":         {"fundamental": elapsed},
    }
    cp.save("03_fundamental", {**state, **updates})
    display.print_node_complete("fundamental", elapsed)
    return updates


def node_risk(state: PipelineState) -> dict:
    """NODE 4: Risk Analysis (runs in parallel with 3 & 5)."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    print(f"\n⚠️  Risk Agent starting...")
    t0 = time.time()

    if cp.should_skip_node("04_risk", state.get("refresh", False)):
        saved = cp.load("04_risk")
        elapsed = time.time() - t0
        display.print_node_complete("risk", elapsed)
        return {
            "risk_complete": saved.get("risk_complete", True),
            "risk_result":   saved.get("risk_result", {}),
            "risk_score":    saved.get("risk_score", 0.0),
            "risk_verdict":  saved.get("risk_verdict", ""),
            "node_timings":  {"risk": elapsed},
        }

    try:
        from agents.risk_agent import RiskAgent
        result = RiskAgent(ticker, config).analyze(refresh=state.get("refresh", False))
    except Exception as e:
        print(f"  [ERROR] Risk agent failed: {e}")
        return {
            "risk_complete": False,
            "risk_result":   {},
            "risk_score":    5.0,
            "risk_verdict":  "ERROR",
            "errors":        [{"node": "risk", "error": str(e)}],
            "node_timings":  {"risk": time.time() - t0},
        }

    display.print_agent_result("risk", result)
    elapsed = time.time() - t0
    updates = {
        "risk_complete": True,
        "risk_result":   result,
        "risk_score":    float(result.get("overall_risk_score", 0)),
        "risk_verdict":  result.get("risk_verdict", ""),
        "node_timings":  {"risk": elapsed},
    }
    cp.save("04_risk", {**state, **updates})
    display.print_node_complete("risk", elapsed)
    return updates


def node_sentiment(state: PipelineState) -> dict:
    """NODE 5: Sentiment Analysis (runs in parallel with 3 & 4)."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    print(f"\n📰 Sentiment Agent starting...")
    t0 = time.time()

    if cp.should_skip_node("05_sentiment", state.get("refresh", False)):
        saved = cp.load("05_sentiment")
        elapsed = time.time() - t0
        display.print_node_complete("sentiment", elapsed)
        return {
            "sentiment_complete": saved.get("sentiment_complete", True),
            "sentiment_result":   saved.get("sentiment_result", {}),
            "sentiment_score":    saved.get("sentiment_score", 0.0),
            "sentiment_verdict":  saved.get("sentiment_verdict", ""),
            "node_timings":       {"sentiment": elapsed},
        }

    try:
        from agents.sentiment_agent import SentimentAgent
        result = SentimentAgent(ticker, config).analyze(refresh=state.get("refresh", False))
    except Exception as e:
        print(f"  [ERROR] Sentiment agent failed: {e}")
        return {
            "sentiment_complete": False,
            "sentiment_result":   {},
            "sentiment_score":    0.0,
            "sentiment_verdict":  "ERROR",
            "errors":             [{"node": "sentiment", "error": str(e)}],
            "node_timings":       {"sentiment": time.time() - t0},
        }

    display.print_agent_result("sentiment", result)
    elapsed = time.time() - t0
    updates = {
        "sentiment_complete": True,
        "sentiment_result":   result,
        "sentiment_score":    float(result.get("weighted_sentiment_score", 0)),
        "sentiment_verdict":  result.get("sentiment_verdict", ""),
        "node_timings":       {"sentiment": elapsed},
    }
    cp.save("05_sentiment", {**state, **updates})
    display.print_node_complete("sentiment", elapsed)
    return updates


def node_lead_analyst(state: PipelineState) -> dict:
    """NODE 6: Lead Analyst Synthesis."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    display.print_node_start("lead")
    t0 = time.time()

    if cp.should_skip_node("06_lead_analyst", state.get("refresh", False)):
        saved = cp.load("06_lead_analyst")
        elapsed = time.time() - t0
        display.print_node_complete("lead", elapsed)
        return {
            "lead_complete":   saved.get("lead_complete", True),
            "memo_result":     saved.get("memo_result", {}),
            "recommendation":  saved.get("recommendation", ""),
            "confidence":      saved.get("confidence", 0),
            "price_target":    saved.get("price_target", 0.0),
            "composite_score": saved.get("composite_score", 0.0),
            "node_timings":    {"lead": elapsed},
        }

    if not all([
        state.get("fundamental_complete"),
        state.get("risk_complete"),
        state.get("sentiment_complete"),
    ]):
        raise ValueError("Cannot run Lead Analyst — prior agents incomplete")

    try:
        from agents.lead_analyst_agent import LeadAnalystAgent
        result = LeadAnalystAgent(ticker, config).analyze(refresh=state.get("refresh", False))
    except Exception as e:
        print(f"  [ERROR] Lead analyst failed: {e}")
        return {
            "lead_complete": False,
            "memo_result":   {},
            "errors":        [{"node": "lead_analyst", "error": str(e)}],
            "node_timings":  {"lead": time.time() - t0},
        }

    display.print_agent_result("lead", result)
    elapsed = time.time() - t0
    updates = {
        "lead_complete":   True,
        "memo_result":     result,
        "recommendation":  result.get("recommendation", ""),
        "confidence":      int(result.get("confidence", 0)),
        "price_target":    float(result.get("price_target_12m", 0)),
        "composite_score": float(result.get("composite_score", 0)),
        "node_timings":    {"lead": elapsed},
    }
    cp.save("06_lead_analyst", {**state, **updates})
    display.print_node_complete("lead", elapsed)
    return updates


def node_generate_pdf(state: PipelineState) -> dict:
    """NODE 7: PDF Generation."""
    ticker, config = _ticker_config(state)
    cp = PipelineCheckpointer(ticker)
    print("\n📄 Generating PDF report...")
    t0 = time.time()

    pdf_path = f"{config.DATA_DIR}/{ticker}/analysis/investment_memo.pdf"

    if (
        not state.get("refresh", False)
        and os.path.exists(pdf_path)
        and cp.should_skip_node("07_pdf", False)
    ):
        elapsed = time.time() - t0
        display.print_node_complete("pdf", elapsed)
        return {
            "pdf_complete": True,
            "pdf_path":     pdf_path,
            "node_timings": {"pdf": elapsed},
        }

    memo_md_path = f"{config.DATA_DIR}/{ticker}/analysis/investment_memo.md"
    memo_json_path = f"{config.DATA_DIR}/{ticker}/analysis/investment_memo.json"

    try:
        with open(memo_md_path, encoding="utf-8") as f:
            memo_md = f.read()
        with open(memo_json_path, encoding="utf-8") as f:
            memo_json = json.load(f)

        from pipeline.pdf_generator import PDFGenerator
        pdf_path = PDFGenerator(ticker).generate(memo_md, memo_json)
        print(f"✅ PDF saved: {pdf_path}")
        print(f"   To open: open {pdf_path}")
    except Exception as e:
        print(f"  [ERROR] PDF generation failed: {e}")
        return {
            "pdf_complete": False,
            "pdf_path":     "",
            "errors":       [{"node": "pdf", "error": str(e)}],
            "node_timings": {"pdf": time.time() - t0},
        }

    elapsed = time.time() - t0
    updates = {
        "pdf_complete": True,
        "pdf_path":     pdf_path,
        "node_timings": {"pdf": elapsed},
    }
    cp.save("07_pdf", {**state, **updates})
    display.print_node_complete("pdf", elapsed)
    return updates
