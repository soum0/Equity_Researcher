from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agents.base_agent import BaseAgent


def normalize_score(score, target_max=10):
    """Normalize LLM score to 0-10 range regardless of what scale LLM used."""
    if score is None:
        return 5.0
    score = float(score)
    if score == 0.0:     # 0.0 means LLM failed, not actually zero quality
        return 5.0
    if score > 10:       # LLM returned 0-100 scale (e.g. 72)
        return round(score / 10, 1)
    elif score <= 1.0:   # LLM returned 0-1 scale (e.g. 0.7)
        return round(score * 10, 1)
    else:                # LLM returned 0-10 scale (e.g. 7.2) ← correct
        return round(score, 1)


def normalize_confidence(conf):
    """Normalize LLM confidence to 0-100 integer regardless of scale."""
    if conf is None:
        return 50
    conf = float(conf)
    if conf <= 1.0:    # 0-1 scale (e.g. 0.8 → 80)
        return int(conf * 100)
    elif conf > 100:   # clamp
        return 95
    else:              # already 0-100
        return int(conf)
from agents.prompts.risk_prompts import (
    RISK_SYSTEM_PROMPT,
    COMPANY_SPECIFIC_RISKS_PROMPT,
    MACRO_RISK_PROMPT,
    RISK_INTERCONNECTION_PROMPT,
    BEAR_CASE_PROMPT,
    RISK_SYNTHESIS_PROMPT,
)

console = Console()

_CACHE_TTL_HOURS = 24

# Macro series display config: (key, label, unit_label, commentary_fn)
_MACRO_SERIES = [
    ("federal_funds_rate",  "Federal Funds Rate",  "%",  lambda v: "HIGH interest rate environment" if v > 4 else "moderate rate environment" if v > 2 else "low rate environment"),
    ("10y_treasury_yield",  "10Y Treasury Yield",  "%",  lambda v: "rising yields compress equity multiples" if v > 4 else "bond yields neutral"),
    ("cpi_inflation",       "CPI Index",           "",   lambda v: ""),
    ("gdp_growth",          "GDP Growth",          "%",  lambda v: "economy growing" if v > 2 else "slow growth" if v > 0 else "contraction"),
    ("unemployment_rate",   "Unemployment",        "%",  lambda v: "labor market healthy" if v < 4.5 else "labor market softening"),
    ("vix_index",           "VIX",                 "",   lambda v: "elevated volatility" if v > 25 else "moderate market volatility" if v > 15 else "low volatility"),
]


class RiskAgent(BaseAgent):
    """
    Risk Analysis Agent.
    Stress-tests the Fundamental Analysis with 4 targeted risk passes
    and a bear case synthesis. Risk score is INVERSE of fundamental score
    (higher = more dangerous).
    """

    # ── Data loaders ──────────────────────────────────────────────────────────

    def load_fundamental_analysis(self) -> dict:
        """Load and extract key fields from fundamental_analysis.json."""
        raw = self.load_result("fundamental_analysis.json")
        if not raw:
            raise FileNotFoundError(
                f"Fundamental analysis not found. Run Step 3 first:\n"
                f"  python main.py {self.ticker} --analyze-fundamentals"
            )

        fin = raw.get("financial_health", {})
        prof = fin.get("profitability", {})
        bs = fin.get("balance_sheet", {})
        cf = fin.get("cash_flow", {})
        val = fin.get("valuation", {})

        # Normalize net_margin: LLM sometimes stores as 27.1, sometimes 0.271
        raw_margin = prof.get("net_margin")
        if raw_margin is not None:
            net_margin = float(raw_margin) if float(raw_margin) >= 1 else float(raw_margin) * 100
        else:
            net_margin = None

        return {
            "full": raw,
            "score": raw.get("overall_score", 0),
            "verdict": raw.get("verdict", "UNKNOWN"),
            "weaknesses": raw.get("key_weaknesses", []),
            "financial_health": fin,
            "competitive_position": raw.get("competitive_position", {}),
            "pe_ratio": val.get("pe_ratio"),
            "net_margin": net_margin,
            "debt_to_equity": bs.get("debt_to_equity"),
            "fcf": raw.get("_fcf_raw"),
        }

    def load_macro_context(self) -> str:
        """Format macro_indicators.json as a readable text block."""
        path = f"{self.config.DATA_DIR}/{self.ticker}/macro_indicators.json"
        if not os.path.exists(path):
            return "Macro data unavailable."

        with open(path) as f:
            macro = json.load(f)

        indicators = macro.get("indicators", {})
        lines = ["Current Macroeconomic Environment:"]

        for key, label, unit, commentary_fn in _MACRO_SERIES:
            ind = indicators.get(key, {})
            if "error" in ind:
                continue
            val = ind.get("latest_value")
            if val is None:
                continue
            trend = ind.get("trend_3m", "")
            date = ind.get("latest_date", "")[:7]
            commentary = commentary_fn(val) if commentary_fn else ""
            unit_str = unit if unit else ""
            comment_str = f" — {commentary}" if commentary else ""
            lines.append(f" {label}: {val}{unit_str} ({trend} trend, {date}){comment_str}")

        return "\n".join(lines)

    # ── Pass 1 ────────────────────────────────────────────────────────────────

    def analyze_company_risks(self, fundamental: dict) -> dict:
        self.logger.info("Pass 1/4: Company-Specific Risk Analysis...")
        console.print("[bold red]  Pass 1/4:[/bold red] Company-Specific Risk Analysis...")

        ctx1 = self.retrieve(
            "risk factors supply chain concentration customer",
            section_filter="Risk Factors",
        )
        ctx2 = self.retrieve("legal proceedings litigation regulatory investigation SEC")
        ctx3 = self.retrieve("cybersecurity data privacy technology risk")
        ctx4 = self.retrieve("debt maturity liquidity covenant financial obligations")

        combined = (
            ctx1["combined_context"] + "\n"
            + ctx2["combined_context"] + "\n"
            + ctx3["combined_context"] + "\n"
            + ctx4["combined_context"]
        )

        weaknesses = fundamental.get("weaknesses", [])
        weaknesses_text = "\n".join(f"{i+1}. {w}" for i, w in enumerate(weaknesses)) or "None identified."

        result = self.llm_call(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_message=COMPANY_SPECIFIC_RISKS_PROMPT.format(
                fundamental_weaknesses=weaknesses_text,
                context=combined,
            ),
        )

        if not result.get("company_specific_risks"):
            self.logger.warning("Pass 1: LLM returned no company_specific_risks.")

        return result

    # ── Pass 2 ────────────────────────────────────────────────────────────────

    def analyze_macro_risks(self, fundamental: dict) -> dict:
        self.logger.info("Pass 2/4: Macroeconomic Risk Analysis...")
        console.print("[bold red]  Pass 2/4:[/bold red] Macroeconomic Risk Analysis...")

        macro_context = self.load_macro_context()

        pe = fundamental.get("pe_ratio") or "N/A"
        margin = fundamental.get("net_margin") or "N/A"
        de = fundamental.get("debt_to_equity") or "N/A"
        fcf = fundamental.get("fcf")
        fcf_str = f"${fcf / 1e9:.1f}B" if fcf and abs(fcf) >= 1e9 else (f"${fcf / 1e6:.0f}M" if fcf else "N/A")

        # Revenue growth from price_data
        rev_growth = "N/A"
        pd_path = f"{self.config.DATA_DIR}/{self.ticker}/price_data.json"
        if os.path.exists(pd_path):
            with open(pd_path) as f:
                pd = json.load(f)
            rev_growth_raw = pd.get("key_ratios", {}).get("growth", {}).get("revenue_growth_yoy")
            if rev_growth_raw is not None:
                rev_growth = f"{rev_growth_raw * 100:.1f}%"

        financial_context = (
            f"Company Financial Profile:\n"
            f" P/E Ratio: {pe}x (vs S&P 500 avg ~22x)\n"
            f" Net Margin: {margin}%\n"
            f" D/E Ratio: {de}x\n"
            f" FCF: {fcf_str}\n"
            f" Revenue Growth YoY: {rev_growth}"
        )

        ctx1 = self.retrieve("interest rate impact debt financing cost")
        ctx2 = self.retrieve("international revenue foreign currency geographic")

        combined_financial = (
            financial_context + "\n\n"
            + ctx1["combined_context"] + "\n"
            + ctx2["combined_context"]
        )

        result = self.llm_call(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_message=MACRO_RISK_PROMPT.format(
                macro_context=macro_context,
                financial_context=combined_financial,
            ),
        )

        if not result.get("macro_risks"):
            self.logger.warning("Pass 2: LLM returned no macro_risks.")

        return result

    # ── Pass 3 ────────────────────────────────────────────────────────────────

    def analyze_risk_interconnections(
        self, company_risks: dict, macro_risks: dict
    ) -> dict:
        self.logger.info("Pass 3/4: Risk Interconnection Analysis (GraphRAG)...")
        console.print("[bold red]  Pass 3/4:[/bold red] Risk Interconnection Analysis (GraphRAG)...")

        all_risk_names = (
            [r.get("name", "") for r in company_risks.get("company_specific_risks", [])]
            + [r.get("name", "") for r in macro_risks.get("macro_risks", [])]
        )

        graph_summaries: list[str] = []
        for risk_name in all_risk_names[:8]:
            if not risk_name:
                continue
            entities = self.retriever.find_entity_by_label(risk_name, node_type="RiskFactor")
            if not entities:
                self.logger.debug(f"No graph entities found for risk: {risk_name}")
                continue
            for entity in entities[:2]:
                subgraph = self.retriever.graph_search(
                    entity["id"],
                    depth=2,
                    edge_types=["HAS_RISK", "IMPACTS", "LINKED_TO", "MENTIONED_IN"],
                )
                summary = subgraph.get("subgraph_summary", "")
                if summary:
                    graph_summaries.append(summary)

        graph_context = "\n".join(graph_summaries) if graph_summaries else "No direct graph matches found for identified risks."

        # Build identified_risks as numbered list
        all_risks_for_prompt: list[str] = []
        for r in company_risks.get("company_specific_risks", []):
            all_risks_for_prompt.append(
                f"{r.get('risk_id', '')} [{r.get('severity', '')}] {r.get('name', '')}: {r.get('description', '')[:120]}"
            )
        for r in macro_risks.get("macro_risks", []):
            all_risks_for_prompt.append(
                f"{r.get('risk_id', '')} [{r.get('severity', '')}] {r.get('name', '')}: {r.get('downside_scenario', '')[:120]}"
            )
        identified_risks = "\n".join(all_risks_for_prompt) or "No risks identified."

        result = self.llm_call(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_message=RISK_INTERCONNECTION_PROMPT.format(
                identified_risks=identified_risks,
                graph_context=graph_context,
            ),
        )

        return result

    # ── Pass 4 ────────────────────────────────────────────────────────────────

    def build_bear_case(
        self,
        fundamental: dict,
        company_risks: dict,
        macro_risks: dict,
        interconnections: dict,
    ) -> dict:
        self.logger.info("Pass 4/4: Building Bear Case...")
        console.print("[bold red]  Pass 4/4:[/bold red] Building Bear Case...")

        # Current price
        current_price = "N/A"
        pd_path = f"{self.config.DATA_DIR}/{self.ticker}/price_data.json"
        if os.path.exists(pd_path):
            with open(pd_path) as f:
                pd = json.load(f)
            current_price = pd.get("realtime_quote", {}).get("current", "N/A")

        # Compact risk summaries (top 3 each)
        def _fmt_company_risks(risks: list, n: int = 3) -> str:
            lines = []
            for r in risks[:n]:
                lines.append(f"- [{r.get('severity','').upper()}] {r.get('name','')}: {r.get('description','')[:100]}")
            return "\n".join(lines) or "None identified."

        def _fmt_macro_risks(risks: list, n: int = 3) -> str:
            lines = []
            for r in risks[:n]:
                lines.append(f"- [{r.get('severity','').upper()}] {r.get('name','')}: {r.get('downside_scenario','')[:100]}")
            return "\n".join(lines) or "None identified."

        def _fmt_chains(chains: list, n: int = 2) -> str:
            lines = []
            for c in chains[:n]:
                lines.append(f"- [{c.get('severity','').upper()}] {c.get('name','')}: {c.get('trigger','')} → {' → '.join(c.get('cascade', [])[:3])}")
            return "\n".join(lines) or "No chains identified."

        company_risks_text = _fmt_company_risks(company_risks.get("company_specific_risks", []))
        macro_risks_text   = _fmt_macro_risks(macro_risks.get("macro_risks", []))
        risk_chains_text   = _fmt_chains(interconnections.get("risk_chains", []))

        result = self.llm_call(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_message=BEAR_CASE_PROMPT.format(
                ticker=self.ticker,
                current_price=current_price,
                pe_ratio=fundamental.get("pe_ratio") or "N/A",
                fundamental_score=fundamental.get("score", 0),
                fundamental_verdict=fundamental.get("verdict", "UNKNOWN"),
                company_risks=company_risks_text,
                macro_risks=macro_risks_text,
                risk_chains=risk_chains_text,
            ),
        )

        # Validate that the downside target is actually below current price
        try:
            price_float = float(current_price)
            downside_raw = result.get("downside_price_target", "")
            downside_float = float(str(downside_raw).replace("$", "").replace(",", ""))
            if downside_float >= price_float:
                corrected = f"${price_float * 0.80:.2f}"
                self.logger.warning(
                    f"Bear case downside corrected: was {downside_raw}, now {corrected}"
                )
                result["downside_price_target"] = corrected
        except (TypeError, ValueError):
            if current_price != "N/A":
                try:
                    result["downside_price_target"] = f"${float(current_price) * 0.80:.2f}"
                except (TypeError, ValueError):
                    pass

        return result

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        fundamental: dict,
        company_risks: dict,
        macro_risks: dict,
        interconnections: dict,
        bear_case: dict,
    ) -> dict:
        self.logger.info("Synthesizing risk analysis...")
        console.print("[bold red]  Synthesis:[/bold red] Producing final risk verdict...")

        n_company = len(company_risks.get("company_specific_risks", []))
        top_company = company_risks.get("top_company_risk", "unknown")
        macro_score = macro_risks.get("macro_environment_score", "N/A")
        recession_resilience = macro_risks.get("recession_resilience", "unknown")
        n_chains = len(interconnections.get("risk_chains", []))
        top_chain = interconnections.get("risk_chains", [{}])[0].get("name", "none") if n_chains else "none"
        bear_title = bear_case.get("bear_case_title", "")
        bear_prob = bear_case.get("downside_scenario_probability", "unknown")

        company_risks_summary = f"{n_company} risks identified; top risk: {top_company}"
        macro_risks_summary = f"Macro env score: {macro_score}/10; recession resilience: {recession_resilience}"
        risk_chains_summary = f"{n_chains} chains identified; most severe: {top_chain}"
        bear_case_summary_text = f'"{bear_title}" — probability: {bear_prob}'

        result = self.llm_call(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_message=RISK_SYNTHESIS_PROMPT.format(
                company_risks_summary=company_risks_summary,
                macro_risks_summary=macro_risks_summary,
                risk_chains_summary=risk_chains_summary,
                bear_case_summary=bear_case_summary_text,
                fundamental_score=fundamental.get("score", 0),
            ),
        )

        return result

    # ── Summary printer ───────────────────────────────────────────────────────

    def print_summary(self, result: dict):
        score = result.get("overall_risk_score", 0)
        rating = result.get("risk_rating", "UNKNOWN")
        verdict = result.get("risk_verdict", "UNKNOWN")
        risk_reward = result.get("synthesis", {}).get("risk_reward_assessment", "N/A")
        top_3 = result.get("top_3_risks", [])
        bear = result.get("bear_case", {})
        generated_at = result.get("_metadata", {}).get("generated_at", "")
        citations = result.get("_metadata", {}).get("citations", [])
        n_chains = len(result.get("risk_interconnections", {}).get("risk_chains", []))
        triggers = result.get("key_triggers", [])
        position = result.get("position_sizing", "N/A")

        # Risk score color: red if high risk, yellow if moderate, green if low
        if isinstance(score, (int, float)) and score >= 7:
            score_color = "red"
        elif isinstance(score, (int, float)) and score >= 5:
            score_color = "yellow"
        else:
            score_color = "green"

        console.print(
            Panel(
                f"[bold]RISK ANALYSIS — {self.ticker}[/bold]\n"
                f"Generated: {generated_at[:19].replace('T', ' ')} UTC",
                box=box.ROUNDED,
                style="bold white",
            )
        )

        console.print(
            f"\n[bold]RISK SCORE:[/bold]    "
            f"[{score_color} bold]{score} / 10  ({rating})[/{score_color} bold]"
        )
        console.print(f"[bold]RISK VERDICT:[/bold]  [bold]{verdict}[/bold]")
        console.print(f"[bold]RISK/REWARD:[/bold]   {risk_reward}\n")

        # Top 3 risks table
        if top_3:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
            table.add_column("TOP 3 RISKS", style="bold", min_width=6)
            table.add_column("NAME", min_width=30)
            table.add_column("SUMMARY", min_width=42)
            for r in top_3:
                rank = f"#{r.get('rank', '')}"
                table.add_row(rank, r.get("name", ""), r.get("one_line", ""))
            console.print(table)

        # Bear case
        bear_title = bear.get("bear_case_title", "N/A")
        bear_prob  = bear.get("downside_scenario_probability", "N/A")
        bear_down  = bear.get("downside_price_target", "N/A")
        console.print(f"[bold]BEAR CASE[/bold]")
        console.print(f"  Title:       {bear_title}")
        console.print(f"  Probability: {bear_prob}")
        console.print(f"  Downside:    {bear_down}\n")

        # Chains + triggers
        console.print(f"[bold]RISK CHAINS:[/bold]   {n_chains} identified")
        if triggers:
            triggers_str = " | ".join(triggers[:3])
            console.print(f"[bold]KEY TRIGGERS:[/bold]  {triggers_str}\n")

        console.print(f"[bold]POSITION SIZING:[/bold] {position}")
        console.print(f"\n[dim]SOURCES CITED: {len(citations)}[/dim]\n")

    # ── Main orchestrator ─────────────────────────────────────────────────────

    def analyze(self, refresh: bool = False) -> dict:
        """Orchestrate all risk passes. Caches result for 24 hours."""
        if not refresh:
            cached = self.load_result("risk_analysis.json")
            if cached:
                meta = cached.get("_metadata", {})
                generated_at = meta.get("generated_at", "")
                if generated_at:
                    try:
                        age = datetime.now(timezone.utc) - datetime.fromisoformat(generated_at)
                        if age < timedelta(hours=_CACHE_TTL_HOURS):
                            console.print(
                                "[dim]Using cached risk analysis "
                                "(run with --refresh to regenerate)[/dim]"
                            )
                            return cached
                    except ValueError:
                        pass

        console.print(f"\n[bold]Risk Analysis — {self.ticker}[/bold]\n")

        fundamental = self.load_fundamental_analysis()

        company_risks    = self.analyze_company_risks(fundamental)
        macro_risks      = self.analyze_macro_risks(fundamental)
        interconnections = self.analyze_risk_interconnections(company_risks, macro_risks)
        bear_case        = self.build_bear_case(fundamental, company_risks, macro_risks, interconnections)
        synthesis        = self.synthesize(fundamental, company_risks, macro_risks, interconnections, bear_case)

        # Normalize risk score
        synthesis["overall_risk_score"] = normalize_score(synthesis.get("overall_risk_score"))

        result = {
            "ticker": self.ticker,
            "analysis_type": "risk",
            "company_specific_risks": company_risks,
            "macro_risks": macro_risks,
            "risk_interconnections": interconnections,
            "bear_case": bear_case,
            "synthesis": synthesis,
            "overall_risk_score": synthesis["overall_risk_score"],
            "risk_rating": synthesis.get("risk_rating"),
            "risk_verdict": synthesis.get("risk_verdict"),
            "top_3_risks": synthesis.get("top_3_risks", []),
            "bear_case_summary": bear_case.get("bear_case_summary"),
            "bear_case_title": bear_case.get("bear_case_title"),
            "downside_price_target": bear_case.get("downside_price_target"),
            "key_triggers": synthesis.get("key_risk_triggers_to_monitor", []),
            "position_sizing": synthesis.get("recommended_position_sizing"),
        }

        self.save_result(result, "risk_analysis.json")
        self.print_summary(result)

        return result
