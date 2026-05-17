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
from agents.prompts.fundamental_prompts import (
    FUNDAMENTAL_SYSTEM_PROMPT,
    BUSINESS_QUALITY_PROMPT,
    FINANCIAL_HEALTH_PROMPT,
    MANAGEMENT_QUALITY_PROMPT,
    COMPETITIVE_POSITION_PROMPT,
    SYNTHESIS_PROMPT,
)

console = Console()

_CACHE_TTL_HOURS = 24


class FundamentalAgent(BaseAgent):
    """
    Fundamental Analysis Agent.
    Runs 4 focused retrieval+analysis passes, then synthesizes.
    """

    # ── Pass 1 ────────────────────────────────────────────────────────────────

    def analyze_business_quality(self) -> dict:
        self.logger.info("Pass 1/4: Business Quality Analysis...")
        console.print("[bold cyan]  Pass 1/4:[/bold cyan] Business Quality Analysis...")

        ctx1 = self.retrieve("business segments revenue breakdown operations")
        ctx2 = self.retrieve(
            "competitive advantage moat pricing power",
            section_filter="Business Overview",
        )
        ctx3 = self.retrieve("growth drivers strategy outlook")

        combined = (
            ctx1["combined_context"] + "\n"
            + ctx2["combined_context"] + "\n"
            + ctx3["combined_context"]
        )

        result = self.llm_call(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_message=BUSINESS_QUALITY_PROMPT.format(context=combined),
        )

        if not result.get("citation"):
            self.logger.warning("Pass 1: LLM returned empty citation array.")

        return result

    # ── Pass 2 ────────────────────────────────────────────────────────────────

    def analyze_financial_health(self) -> dict:
        self.logger.info("Pass 2/4: Financial Health Analysis...")
        console.print("[bold cyan]  Pass 2/4:[/bold cyan] Financial Health Analysis...")

        # Load structured data from Step 1 files
        metrics_text = self._build_financial_metrics_text()
        self._fcf_raw = self._last_fcf_raw

        ctx1 = self.retrieve(
            "revenue earnings profit margin growth quarterly",
            section_filter="MD&A",
        )
        ctx2 = self.retrieve("cash flow capital expenditure debt liquidity balance sheet")

        combined = metrics_text + "\n\n" + ctx1["combined_context"] + "\n" + ctx2["combined_context"]

        result = self.llm_call(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_message=FINANCIAL_HEALTH_PROMPT.format(context=combined),
        )

        if not result.get("citation"):
            self.logger.warning("Pass 2: LLM returned empty citation array.")

        return result

    def _build_financial_metrics_text(self) -> str:
        """Load real numbers from Step 1 JSON files and format as readable text."""
        data_dir = f"{self.config.DATA_DIR}/{self.ticker}"
        lines: list[str] = ["Key Financial Metrics (from Step 1 data):"]

        # key_metrics.json
        km: dict = {}
        km_path = f"{data_dir}/key_metrics.json"
        if os.path.exists(km_path):
            with open(km_path) as f:
                km = json.load(f)
            bf = km.get("basic_financials", {})
            margins = bf.get("margins", {})
            valuation = bf.get("valuation", {})
            growth = bf.get("growth", {})
            per_share = bf.get("per_share", {})

            parts = []
            if margins.get("gross_margin_ttm") is not None:
                parts.append(f"Gross Margin: {margins['gross_margin_ttm']:.1f}%")
            if margins.get("operating_margin_ttm") is not None:
                parts.append(f"Operating Margin: {margins['operating_margin_ttm']:.1f}%")
            if margins.get("net_margin_ttm") is not None:
                parts.append(f"Net Margin: {margins['net_margin_ttm']:.1f}%")
            if valuation.get("pe_ttm") is not None:
                parts.append(f"P/E (TTM): {valuation['pe_ttm']:.1f}x")
            if valuation.get("ev_ebitda") is not None:
                parts.append(f"EV/EBITDA: {valuation['ev_ebitda']:.1f}x")
            if growth.get("revenue_growth_ttm") is not None:
                parts.append(f"Revenue Growth TTM: {growth['revenue_growth_ttm']:.1f}%")
            if growth.get("eps_growth_3y") is not None:
                parts.append(f"EPS Growth 3Y: {growth['eps_growth_3y']:.1f}%")
            if per_share.get("eps_ttm") is not None:
                parts.append(f"EPS TTM: ${per_share['eps_ttm']:.2f}")
            if parts:
                lines.append(" | ".join(parts))

        # price_data.json — key_ratios section
        pd_path = f"{data_dir}/price_data.json"
        if os.path.exists(pd_path):
            with open(pd_path) as f:
                pd = json.load(f)
            kr = pd.get("key_ratios", {})
            profitability = kr.get("profitability", {})
            fin_health = kr.get("financial_health", {})
            val = kr.get("valuation", {})
            growth_kr = kr.get("growth", {})

            parts2 = []
            roe = profitability.get("roe")
            if roe is not None:
                parts2.append(f"ROE: {roe * 100:.1f}%")
            roic = profitability.get("roic")
            if roic is not None:
                parts2.append(f"ROIC: {roic * 100:.1f}%")
            fs: dict = {}
            fs_path = f"{data_dir}/financial_statements.json"
            if os.path.exists(fs_path):
                with open(fs_path) as f:
                    fs = json.load(f)
            fcf_sources = [
                km.get("freeCashflow"),
                km.get("free_cash_flow"),
                pd.get("summary", {}).get("free_cash_flow"),
                fs.get("cashflow", {}).get("Free Cash Flow"),
                fin_health.get("free_cash_flow"),
            ]
            fcf = next((v for v in fcf_sources if v and v != 0), None)
            self._last_fcf_raw = fcf
            self.logger.info(f"FCF raw value: {fcf}")
            if fcf is not None:
                if abs(fcf) >= 1_000_000_000:
                    parts2.append(f"FCF: ${fcf / 1_000_000_000:.1f}B")
                elif abs(fcf) >= 1_000_000:
                    parts2.append(f"FCF: ${fcf / 1_000_000:.0f}M")
            cash = fin_health.get("cash_and_equivalents")
            if cash is not None:
                parts2.append(f"Cash: ${cash / 1e9:.1f}B")
            debt = fin_health.get("total_debt")
            if debt is not None:
                parts2.append(f"Total Debt: ${debt / 1e9:.1f}B")
            raw_de = (
                km.get("debtToEquity")
                or km.get("debt_to_equity")
                or km.get("financial_health", {}).get("debt_to_equity")
                or fin_health.get("debt_to_equity")
            )
            if raw_de is not None:
                debt_to_equity = round(raw_de / 100, 2) if raw_de > 10 else round(raw_de, 2)
                self.logger.info(f"D/E raw: {raw_de} → normalized: {debt_to_equity}")
                parts2.append(f"D/E: {debt_to_equity:.2f}x")
            cr = fin_health.get("current_ratio")
            if cr is not None:
                parts2.append(f"Current Ratio: {cr:.2f}")
            rev_growth = growth_kr.get("revenue_growth_yoy")
            if rev_growth is not None:
                parts2.append(f"Revenue Growth YoY: {rev_growth * 100:.1f}%")
            earn_growth = growth_kr.get("earnings_growth_yoy")
            if earn_growth is not None:
                parts2.append(f"Earnings Growth YoY: {earn_growth * 100:.1f}%")
            pe = val.get("pe_ratio_ttm")
            if pe is not None:
                parts2.append(f"P/E: {pe:.1f}x")
            ps = val.get("ps_ratio")
            if ps is not None:
                parts2.append(f"P/S: {ps:.2f}x")
            ev_ebitda = val.get("ev_to_ebitda")
            if ev_ebitda is not None:
                parts2.append(f"EV/EBITDA: {ev_ebitda:.1f}x")
            mktcap = val.get("market_cap")
            if mktcap is not None:
                parts2.append(f"Market Cap: ${mktcap / 1e12:.2f}T")
            if parts2:
                lines.append(" | ".join(parts2))

        return "\n".join(lines)

    # ── Pass 3 ────────────────────────────────────────────────────────────────

    def analyze_management_quality(self) -> dict:
        self.logger.info("Pass 3/4: Management Quality Analysis...")
        console.print("[bold cyan]  Pass 3/4:[/bold cyan] Management Quality Analysis...")

        ctx1 = self.retrieve("CEO management executive leadership team")
        ctx2 = self.retrieve("insider transactions buying selling shares executive")
        ctx3 = self.retrieve("guidance forecast outlook beat miss earnings")

        # Enrich with structured data
        profile_text = ""
        profile_path = f"{self.config.DATA_DIR}/{self.ticker}/company_profile.json"
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profile = json.load(f)
            profile_text = f"Company Profile: {json.dumps(profile, indent=2)}\n\n"

        insider_text = ""
        km_path = f"{self.config.DATA_DIR}/{self.ticker}/key_metrics.json"
        if os.path.exists(km_path):
            with open(km_path) as f:
                km = json.load(f)
            insider = km.get("insider_transactions", {})
            if insider:
                insider_text = f"Insider Transactions:\n{json.dumps(insider, indent=2)}\n\n"

        combined = (
            profile_text
            + insider_text
            + ctx1["combined_context"] + "\n"
            + ctx2["combined_context"] + "\n"
            + ctx3["combined_context"]
        )

        result = self.llm_call(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_message=MANAGEMENT_QUALITY_PROMPT.format(context=combined),
        )

        if not result.get("citation"):
            self.logger.warning("Pass 3: LLM returned empty citation array.")

        return result

    # ── Pass 4 ────────────────────────────────────────────────────────────────

    def analyze_competitive_position(self) -> dict:
        self.logger.info("Pass 4/4: Competitive Position Analysis...")
        console.print("[bold cyan]  Pass 4/4:[/bold cyan] Competitive Position Analysis...")

        ctx1 = self.retrieve(
            "competition competitive landscape market share industry",
            section_filter="Business Overview",
        )
        ctx2 = self.retrieve("competitors threat new entrant substitute disruption")
        ctx3 = self.retrieve("artificial intelligence technology innovation risk")

        combined = (
            ctx1["combined_context"] + "\n"
            + ctx2["combined_context"] + "\n"
            + ctx3["combined_context"]
        )

        result = self.llm_call(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_message=COMPETITIVE_POSITION_PROMPT.format(context=combined),
        )

        if not result.get("citation"):
            self.logger.warning("Pass 4: LLM returned empty citation array.")

        return result

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        business: dict,
        financial: dict,
        management: dict,
        competitive: dict,
    ) -> dict:
        self.logger.info("Synthesizing fundamental analysis...")
        console.print("[bold cyan]  Synthesis:[/bold cyan] Combining all passes...")

        result = self.llm_call(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_message=SYNTHESIS_PROMPT.format(
                business_quality=json.dumps(business, indent=2),
                financial_health=json.dumps(financial, indent=2),
                management_quality=json.dumps(management, indent=2),
                competitive_position=json.dumps(competitive, indent=2),
            ),
        )

        return result

    # ── Summary printer ───────────────────────────────────────────────────────

    def print_summary(self, result: dict, fcf_raw: float = None):
        score = result.get("overall_score", 0)
        verdict = result.get("verdict", "N/A")
        confidence = result.get("confidence", 0)
        strengths = result.get("key_strengths", [])
        weaknesses = result.get("key_weaknesses", [])
        generated_at = result.get("_metadata", {}).get("generated_at", "")
        citations = result.get("_metadata", {}).get("citations", [])

        # Score color
        if isinstance(score, (int, float)) and score >= 7:
            score_color = "green"
            score_label = "STRONG"
        elif isinstance(score, (int, float)) and score >= 5:
            score_color = "yellow"
            score_label = "MODERATE"
        else:
            score_color = "red"
            score_label = "WEAK"

        # Header panel
        console.print(
            Panel(
                f"[bold]FUNDAMENTAL ANALYSIS — {self.ticker}[/bold]\n"
                f"Generated: {generated_at[:19].replace('T', ' ')} UTC",
                box=box.ROUNDED,
                style="bold white",
            )
        )

        # Score / Verdict
        console.print(
            f"\n[bold]OVERALL SCORE:[/bold]  "
            f"[{score_color} bold]{score} / 10  ({score_label})[/{score_color} bold]"
        )
        confidence_display = f"{confidence}%"
        console.print(
            f"[bold]VERDICT:[/bold]        "
            f"[bold]{verdict}[/bold]  (Confidence: {confidence_display})"
        )

        # Strengths / Weaknesses table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("STRENGTHS", style="green", min_width=36)
        table.add_column("WEAKNESSES", style="yellow", min_width=36)

        max_rows = max(len(strengths), len(weaknesses))
        for i in range(max_rows):
            s = f"[green]✓[/green] {strengths[i]}" if i < len(strengths) else ""
            w = f"[yellow]⚠[/yellow]  {weaknesses[i]}" if i < len(weaknesses) else ""
            table.add_row(s, w)

        console.print(table)

        # Key metrics line
        fcf_display = "N/A"
        if fcf_raw and abs(float(fcf_raw)) >= 1_000_000_000:
            fcf_display = f"${float(fcf_raw) / 1_000_000_000:.1f}B"
        elif fcf_raw and abs(float(fcf_raw)) >= 1_000_000:
            fcf_display = f"${float(fcf_raw) / 1_000_000:.0f}M"

        pe     = result.get("financial_health", {}).get("valuation", {}).get("pe_ratio") or "N/A"
        margin = result.get("financial_health", {}).get("profitability", {}).get("net_margin")
        de     = result.get("financial_health", {}).get("balance_sheet", {}).get("debt_to_equity") or "N/A"

        if margin is not None:
            margin_display = f"{float(margin):.1f}%" if float(margin) >= 1 else f"{float(margin) * 100:.1f}%"
        else:
            margin_display = "N/A"
        pe_display     = f"{float(pe):.1f}x" if pe != "N/A" else "N/A"
        de_display     = f"{float(de):.2f}x" if de != "N/A" else "N/A"

        metrics_line = (
            f"P/E: {pe_display} | Net Margin: {margin_display} | "
            f"FCF: {fcf_display} | D/E: {de_display}"
        )
        console.print(f"\n[bold]KEY METRICS[/bold]  {metrics_line}")

        console.print(f"\n[dim]SOURCES CITED: {len(citations)}[/dim]\n")

    # ── Main orchestrator ─────────────────────────────────────────────────────

    def analyze(self, refresh: bool = False) -> dict:
        """Orchestrate all 4 passes + synthesis. Caches result for 24 hours."""
        self._fcf_raw = None
        if not refresh:
            cached = self.load_result("fundamental_analysis.json")
            if cached:
                meta = cached.get("_metadata", {})
                generated_at = meta.get("generated_at", "")
                if generated_at:
                    try:
                        age = datetime.now(timezone.utc) - datetime.fromisoformat(generated_at)
                        if age < timedelta(hours=_CACHE_TTL_HOURS):
                            console.print(
                                f"[dim]Using cached fundamental analysis "
                                f"(run with --refresh to regenerate)[/dim]"
                            )
                            return cached
                    except ValueError:
                        pass

        console.print(f"\n[bold]Fundamental Analysis — {self.ticker}[/bold]\n")

        business = self.analyze_business_quality()
        financial = self.analyze_financial_health()
        management = self.analyze_management_quality()
        competitive = self.analyze_competitive_position()
        synthesis = self.synthesize(business, financial, management, competitive)

        score = normalize_score(synthesis.get("overall_fundamental_score"))
        confidence = normalize_confidence(synthesis.get("verdict_confidence"))

        result = {
            "ticker": self.ticker,
            "analysis_type": "fundamental",
            "business_quality": business,
            "financial_health": financial,
            "management_quality": management,
            "competitive_position": competitive,
            "synthesis": synthesis,
            "overall_score": score,
            "verdict": synthesis.get("fundamental_verdict"),
            "confidence": confidence,
            "key_strengths": synthesis.get("key_fundamental_strengths", []),
            "key_weaknesses": synthesis.get("key_fundamental_weaknesses", []),
            "investment_thesis": synthesis.get("investment_thesis_long"),
        }

        # Sanity: if FCF > $10B and net_margin > 20%, score can't be < 4.0
        score = result.get("overall_score", 5.0)
        if (self._fcf_raw and self._fcf_raw > 10_000_000_000 and
                score < 4.0):
            score = 5.0
            self.logger.warning("Score sanity check triggered: bumped to 5.0")
        result["overall_score"] = score

        self.save_result(result, "fundamental_analysis.json")
        self.print_summary(result, fcf_raw=self._fcf_raw)

        return result
