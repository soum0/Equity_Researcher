from __future__ import annotations

from datetime import datetime, timezone

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from pipeline.state import PipelineState

console = Console()


class PipelineDisplay:
    """Manages Rich terminal display during pipeline execution."""

    NODE_LABELS = {
        "ingest":      "📥 Data Ingestion",
        "build_graph": "🕸️  GraphRAG Build",
        "fundamental": "🔍 Fundamental Analysis",
        "risk":        "⚠️  Risk Analysis",
        "sentiment":   "📰 Sentiment Analysis",
        "lead":        "🎯 Lead Analyst",
        "pdf":         "📄 PDF Generation",
    }

    def print_pipeline_header(self, ticker: str, company_name: str = ""):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        title_line = f"🚀 AI Financial Analysis Pipeline"
        ticker_line = f"Ticker: {ticker}"
        if company_name:
            ticker_line += f"  |  {company_name}"
        time_line = f"Started: {now}"

        console.print()
        console.print(Panel(
            f"{title_line}\n{ticker_line}\n{time_line}",
            box=box.DOUBLE,
            border_style="blue bold",
            padding=(0, 2),
        ))
        console.print()
        console.print(
            "  Pipeline: [Ingest] → [Graph] → "
            "[[bold]Fund + Risk + Sent[/bold] (parallel)] → [Lead] → [PDF]"
        )
        console.print()

    def print_node_start(self, node_name: str):
        label = self.NODE_LABELS.get(node_name, node_name)
        console.rule(f"[bold]🔄 Starting: {label}[/bold]", style="cyan")

    def print_ingestion_results(self, ingested_files: list[dict]):
        table = Table(
            title="📥 DATA INGESTION COMPLETE",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            title_style="bold green",
        )
        table.add_column("Type", style="bold", width=10)
        table.add_column("File", width=32)
        table.add_column("Records / Size", justify="right", width=18)

        sec_total_chars = 0
        sec_count = 0
        for entry in ingested_files:
            ftype = entry.get("type", "")
            fname = entry.get("file", "")
            if ftype == "price":
                detail = f"{entry.get('records', 0)} days"
            elif ftype == "news":
                detail = f"{entry.get('records', 0)} articles"
            elif ftype == "sec":
                chars = entry.get("chars", 0)
                sec_total_chars += chars
                sec_count += 1
                detail = f"{chars:,} chars"
            elif ftype == "macro":
                detail = f"{entry.get('records', 0)} FRED series"
            elif ftype == "profile":
                detail = entry.get("company", "")
            else:
                detail = str(entry.get("records", ""))
            table.add_row(ftype.title(), fname, detail)

        console.print(table)
        if sec_count:
            console.print(
                f"  Total SEC text: [bold]{sec_total_chars:,}[/bold] chars across {sec_count} filings"
            )
        console.print()

    def print_graph_results(self, graph_stats: dict):
        table = Table(
            title="🕸️  KNOWLEDGE GRAPH BUILT",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            title_style="bold green",
        )
        table.add_column("Metric", width=30)
        table.add_column("Value", justify="right", width=10)

        chunking = graph_stats.get("chunking", {})
        extraction = graph_stats.get("extraction", {})
        vector = graph_stats.get("vector_store", {})
        graph = graph_stats.get("graph", {})

        rows = [
            ("Total Chunks",         chunking.get("total_chunks", 0)),
            ("Entities Extracted",   extraction.get("total_entities", 0)),
            ("Relationships Mapped", extraction.get("total_relationships", 0)),
            ("Graph Nodes",          graph.get("total_nodes", 0)),
            ("Graph Edges",          graph.get("total_edges", 0)),
            ("Vectors Indexed",      vector.get("total_vectors", 0)),
        ]
        for metric, value in rows:
            table.add_row(metric, str(value))
        console.print(table)

        # Node type breakdown as ASCII bars
        node_types = graph_stats.get("node_type_breakdown", {})
        if node_types:
            console.print("  [bold]NODE TYPE BREAKDOWN[/bold]")
            max_count = max(node_types.values(), default=1)
            total = sum(node_types.values())
            for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
                bar = "█" * int(count / max_count * 20)
                pct = int(count / total * 100) if total else 0
                console.print(f"  {ntype:<14} {bar:<20}  {count:>3}  ({pct}%)")
            console.print()

        # Top connected nodes
        top_nodes = graph_stats.get("top_connected_nodes", [])
        if top_nodes:
            console.print("  [bold]TOP CONNECTED NODES[/bold]")
            for i, node in enumerate(top_nodes[:3], 1):
                console.print(
                    f"  #{i}  {node.get('label', '')} "
                    f"({node.get('type', '')}) — {node.get('degree', 0)} connections"
                )
        console.print()

    def print_agent_result(self, agent_name: str, result: dict):
        if agent_name == "fundamental":
            self._print_fundamental(result)
        elif agent_name == "risk":
            self._print_risk(result)
        elif agent_name == "sentiment":
            self._print_sentiment(result)
        elif agent_name == "lead":
            self._print_lead(result)

    def _print_fundamental(self, result: dict):
        score = result.get("overall_score", 0)
        verdict = result.get("verdict", "")
        confidence = result.get("confidence", 0)
        strengths = result.get("key_strengths", [])[:3]
        weaknesses = result.get("key_weaknesses", [])[:3]
        fh = result.get("financial_health", {})
        prof = fh.get("profitability", {})
        cf = fh.get("cash_flow", {})

        score_label = "STRONG" if score >= 7 else ("MODERATE" if score >= 5 else "WEAK")
        table = Table(
            title=f"🔍 FUNDAMENTAL ANALYSIS COMPLETE",
            box=box.SIMPLE_HEAVY,
            show_header=False,
            title_style="bold green",
            padding=(0, 1),
        )
        table.add_column("Left", width=35)
        table.add_column("Right", width=35)
        table.add_row(
            f"[bold]Score: {score}/10 ({score_label})[/bold]",
            f"[bold]Verdict: {verdict}[/bold]  (Confidence: {confidence}%)",
        )
        # Strengths vs weaknesses side by side
        max_rows = max(len(strengths), len(weaknesses))
        for i in range(max_rows):
            s = f"[green]✓ {strengths[i]}[/green]" if i < len(strengths) else ""
            w = f"[yellow]⚠ {weaknesses[i]}[/yellow]" if i < len(weaknesses) else ""
            table.add_row(s, w)
        # Financial metrics row
        pe = result.get("financial_health", {}).get("valuation", {}).get("pe_ratio", "N/A")
        margin = prof.get("net_margin", "N/A")
        fcf = cf.get("free_cash_flow", "N/A")
        table.add_row(
            f"P/E: {pe}x | Margin: {margin}%",
            f"FCF: ${fcf}B",
        )
        console.print(table)
        console.print()

    def _print_risk(self, result: dict):
        score = result.get("overall_risk_score", 0)
        rating = result.get("risk_rating", "")
        verdict = result.get("risk_verdict", "")
        risks = result.get("top_3_risks", [])
        bear_title = result.get("bear_case_title", "")
        downside = result.get("downside_price_target", "N/A")

        table = Table(
            title="⚠️  RISK ANALYSIS COMPLETE",
            box=box.SIMPLE_HEAVY,
            show_header=False,
            title_style="bold red",
            padding=(0, 1),
        )
        table.add_column("Field", width=72)
        table.add_row(f"[bold]Risk Score: {score}/10 ({rating})[/bold]  |  Verdict: {verdict}")
        for r in risks:
            rank = r.get("rank", "")
            name = r.get("name", "")
            table.add_row(f"  #{rank} {name}")
        table.add_row(f'  Bear Case: "{bear_title[:60]}"')
        table.add_row(f"  Downside Target: {downside}")
        console.print(table)
        console.print()

    def _print_sentiment(self, result: dict):
        score = result.get("weighted_sentiment_score", 0)
        verdict = result.get("sentiment_verdict", "")
        breakdown = result.get("article_breakdown", {})
        total = result.get("total_articles", 0)
        bullish = breakdown.get("bullish", 0)
        bearish = breakdown.get("bearish", 0)
        neutral = breakdown.get("neutral", 0)
        themes = result.get("dominant_themes", [])
        key_event = result.get("most_important_event", "")

        score_label = "LEANING BULLISH" if score > 0.1 else ("LEANING BEARISH" if score < -0.1 else "NEUTRAL")
        table = Table(
            title="📰 SENTIMENT ANALYSIS COMPLETE",
            box=box.SIMPLE_HEAVY,
            show_header=False,
            title_style="bold blue",
            padding=(0, 1),
        )
        table.add_column("Field", width=72)
        table.add_row(f"[bold]Score: {score:+.2f} ({score_label})[/bold]  |  Verdict: {verdict}")
        if total:
            bull_bar = "█" * int(bullish / total * 20) if total else ""
            bear_bar = "█" * int(bearish / total * 20) if total else ""
            neut_bar = "█" * int(neutral / total * 20) if total else ""
            table.add_row(f"  [green]Bullish: {bullish:>3} ({int(bullish/total*100)}%) {bull_bar}[/green]")
            table.add_row(f"  [dim]Neutral: {neutral:>3} ({int(neutral/total*100)}%) {neut_bar}[/dim]")
            table.add_row(f"  [red]Bearish: {bearish:>3} ({int(bearish/total*100)}%) {bear_bar}[/red]")
        if themes:
            top = themes[0]
            table.add_row(
                f"  Top Theme: {top.get('theme', '')} ({top.get('sentiment', '')}, {top.get('intensity', '')})"
            )
        if key_event:
            table.add_row(f'  Key Event: "{key_event[:65]}"')
        console.print(table)
        console.print()

    def _print_lead(self, result: dict):
        rec = result.get("recommendation", "HOLD")
        confidence = result.get("confidence", 0)
        price_target = result.get("price_target_12m", 0)
        composite = result.get("composite_score", 0)
        ret_pct = result.get("expected_return_pct", 0)
        color = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(rec, "white")
        console.print(Panel(
            f"[bold {color}]{rec}[/bold {color}]  |  "
            f"${price_target}  |  {confidence}% conf  |  "
            f"Composite: {composite:.3f}  ({ret_pct:+.1f}%)",
            title="🎯 LEAD ANALYST COMPLETE",
            box=box.SIMPLE_HEAVY,
            border_style=color,
        ))
        console.print()

    def print_parallel_start(self):
        console.print(Panel(
            "⚡ Running 3 agents in parallel...\n"
            "🔍 Fundamental  ⚠️  Risk  📰 Sentiment",
            box=box.SIMPLE_HEAVY,
            border_style="yellow",
            padding=(0, 2),
        ))
        console.print()

    def print_node_complete(self, node_name: str, elapsed: float):
        label = self.NODE_LABELS.get(node_name, node_name)
        mins = int(elapsed // 60)
        secs = elapsed % 60
        time_str = f"{mins}m {secs:.0f}s" if mins else f"{secs:.1f}s"
        console.print(f"[green]✅ {label}[/green] — completed in {time_str}")

    def print_final_summary(self, state: "PipelineState"):
        ticker = state.get("ticker", "")
        timings = state.get("node_timings", {})
        total = sum(timings.values())
        total_m = int(total // 60)
        total_s = int(total % 60)

        console.print()
        console.rule(f"[bold]🏁 PIPELINE COMPLETE — {ticker}[/bold]", style="green")
        console.print(f"  Total Time: {total_m}m {total_s}s")
        console.rule(style="green")
        console.print()

        # Node timings
        console.print("[bold]NODE TIMINGS[/bold]")
        node_order = [
            ("ingest",      "📥 Ingestion"),
            ("build_graph", "🕸️  Graph Build"),
            ("fundamental", "🔍 Fundamental"),
            ("risk",        "⚠️  Risk"),
            ("sentiment",   "📰 Sentiment"),
            ("lead",        "🎯 Lead Analyst"),
            ("pdf",         "📄 PDF"),
        ]
        parallel_nodes = {"fundamental", "risk", "sentiment"}
        for key, label in node_order:
            if key in timings:
                t = timings[key]
                m = int(t // 60)
                s = t % 60
                time_str = f"{m}m {s:.0f}s" if m else f"{s:.0f}s"
                parallel_marker = " ┐" if key == "fundamental" else (
                    " ├ parallel" if key == "risk" else (
                        " ┘" if key == "sentiment" else ""
                    )
                )
                console.print(f"  {label:<22} {time_str:>8}{parallel_marker}")
        console.print()

        # Recommendation box
        rec = state.get("recommendation", "")
        confidence = state.get("confidence", 0)
        price_target = state.get("price_target", 0)
        composite = state.get("composite_score", 0)
        ret_pct = state.get("memo_result", {}).get("expected_return_pct", 0)
        color = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(rec, "white")
        if rec:
            console.print(Panel(
                f"[bold {color}]  {rec}  [/bold {color}]  "
                f"Confidence: {confidence}%\n"
                f"  Price Target: ${price_target}  ({ret_pct:+.1f}%)\n"
                f"  Composite:    {composite:.3f} / 1.00",
                box=box.DOUBLE,
                border_style=color,
                padding=(0, 2),
            ))
            console.print()

        # Scorecard
        console.print("[bold]SCORECARD[/bold]")
        console.print(f"  Fundamental: {state.get('fundamental_score', 'N/A')}/10  →  {state.get('fundamental_verdict', '')}")
        console.print(f"  Risk:        {state.get('risk_score', 'N/A')}/10  →  {state.get('risk_verdict', '')}")
        console.print(f"  Sentiment:  {state.get('sentiment_score', 0.0):+.2f}    →  {state.get('sentiment_verdict', '')}")
        console.print()

        # Output files
        base = f"data/{ticker}/analysis"
        pdf_path = state.get("pdf_path", "")
        console.print("[bold]OUTPUT FILES[/bold]")
        console.print(f"  📄 {base}/investment_memo.json")
        console.print(f"  📝 {base}/investment_memo.md")
        if pdf_path:
            console.print(f"  🖨️  {pdf_path}  ← download this")
            console.print(f"\n  To open PDF: [bold]open {pdf_path}[/bold]")
        console.print()
