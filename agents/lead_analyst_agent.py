import json
import os
from datetime import datetime, timezone

from openai import OpenAI
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from agents.prompts.lead_analyst_prompts import (
    BULL_BEAR_CASE_PROMPT,
    DISAGREEMENT_RESOLUTION_PROMPT,
    INVESTMENT_MEMO_PROMPT,
    LEAD_ANALYST_SYSTEM_PROMPT,
)
from config import Config
from ingestion.utils import setup_logger

console = Console()


class LeadAnalystAgent:
    """
    Lead Analyst Agent — synthesizes all prior analyses into the
    final investment recommendation and memo.

    Makes 3 LLM calls:
      1. Disagreement resolution
      2. Bull/bear case writing
      3. Investment memo writing

    Plus quantitative scoring (pure Python math, no LLM).
    """

    def __init__(self, ticker: str, config: Config | None = None):
        self.ticker = ticker.upper()
        self.config = config or Config()
        self.logger = setup_logger(self.__class__.__name__)
        self.llm_client = OpenAI(
            base_url=self.config.AGENT_BASE_URL,
            api_key=self.config.AGENT_API_KEY,
        )
        self.citations: list[str] = []

    def llm_call(self, system_prompt: str, user_message: str) -> dict:
        """Call the LLM and return parsed JSON."""
        import re
        import time

        full_system = system_prompt + "\n\nReturn ONLY valid JSON. No markdown, no explanation."

        max_chars = 8000 * 4
        if len(full_system) + len(user_message) > max_chars:
            allowed = max_chars - len(full_system) - 200
            if allowed > 500:
                user_message = user_message[:allowed] + "\n\n[Context truncated for length]"
                self.logger.warning(f"Prompt truncated to {allowed} chars")

        time.sleep(0.5)
        start = time.time()
        response = self.llm_client.chat.completions.create(
            model=self.config.AGENT_MODEL,
            max_tokens=self.config.AGENT_MAX_TOKENS,
            temperature=self.config.AGENT_TEMPERATURE,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_message},
            ],
        )
        latency = time.time() - start
        usage = response.usage
        self.logger.info(
            f"LLM call | model={self.config.AGENT_MODEL} | "
            f"prompt_tokens={usage.prompt_tokens} | "
            f"completion_tokens={usage.completion_tokens} | "
            f"latency={latency:.1f}s"
        )

        raw = response.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning(f"Failed to parse LLM response as JSON. Raw preview: {raw[:300]}")
        return {}

    def save_result(self, result: dict, filename: str):
        result["_metadata"] = {
            "ticker": self.ticker,
            "agent": self.__class__.__name__,
            "model": self.config.AGENT_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "citations": self.citations,
        }
        out_dir = f"{self.config.DATA_DIR}/{self.ticker}/analysis"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Saved: {path}")

    def load_result(self, filename: str) -> dict | None:
        path = f"{self.config.DATA_DIR}/{self.ticker}/analysis/{filename}"
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_analyses(self) -> dict:
        """
        Load all three agent outputs + Step 1 data.
        Raises FileNotFoundError if fundamental_analysis.json or price_data.json missing.
        """
        base = f"{self.config.DATA_DIR}/{self.ticker}"

        fundamental_path = f"{base}/analysis/fundamental_analysis.json"
        if not os.path.exists(fundamental_path):
            raise FileNotFoundError(
                f"fundamental_analysis.json not found. Run --analyze-fundamentals first."
            )

        price_path = f"{base}/price_data.json"
        if not os.path.exists(price_path):
            raise FileNotFoundError(
                f"price_data.json not found. Run Step 1 ingestion first."
            )

        risk_path = f"{base}/analysis/risk_analysis.json"
        if not os.path.exists(risk_path):
            self.logger.warning("risk_analysis.json missing — using defaults (risk_score=5.0)")

        sentiment_path = f"{base}/analysis/sentiment_analysis.json"
        if not os.path.exists(sentiment_path):
            self.logger.warning("sentiment_analysis.json missing — using defaults (sentiment_score=0.0)")

        return {
            "fundamental": self._load_json(fundamental_path),
            "risk": self._load_json(risk_path),
            "sentiment": self._load_json(sentiment_path),
            "price_data": self._load_json(price_path),
            "profile": self._load_json(f"{base}/company_profile.json"),
            "metrics": self._load_json(f"{base}/key_metrics.json"),
        }

    def compute_composite_score(self, analyses: dict) -> dict:
        """Compute the quantitative composite score. Pure math — no LLM."""
        fundamental_score = float(analyses["fundamental"].get("overall_score", 5.0))
        risk_score = float(analyses["risk"].get("overall_risk_score", 5.0) if analyses["risk"] else 5.0)
        sentiment_score = float(analyses["sentiment"].get("weighted_sentiment_score", 0.0) if analyses["sentiment"] else 0.0)

        incomplete_data = not analyses["risk"] or not analyses["sentiment"]

        fundamental_component = fundamental_score / 10.0
        risk_component = 1.0 - (risk_score / 10.0)
        sentiment_component = (sentiment_score + 1.0) / 2.0

        composite = (
            fundamental_component * 0.40
            + risk_component * 0.35
            + sentiment_component * 0.25
        )

        if composite >= 0.65:
            recommendation = "BUY"
        elif composite >= 0.45:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"

        base_confidence = int(composite * 100)

        fundamental_verdict = analyses["fundamental"].get("verdict", "")
        risk_verdict = analyses["risk"].get("risk_verdict", "") if analyses["risk"] else ""
        sentiment_verdict = analyses["sentiment"].get("sentiment_verdict", "") if analyses["sentiment"] else ""

        fund_agrees = (
            ("BUY" in fundamental_verdict and recommendation == "BUY") or
            ("HOLD" in fundamental_verdict and recommendation == "HOLD") or
            ("SELL" in fundamental_verdict and recommendation == "SELL")
        )
        risk_agrees = (
            (risk_verdict == "PASS" and recommendation in ["BUY", "HOLD"]) or
            (risk_verdict == "CAUTION" and recommendation == "HOLD") or
            (risk_verdict == "FAIL" and recommendation == "SELL")
        )
        sent_agrees = (
            (sentiment_verdict == "POSITIVE" and recommendation == "BUY") or
            (sentiment_verdict == "NEUTRAL" and recommendation == "HOLD") or
            (sentiment_verdict == "NEGATIVE" and recommendation == "SELL")
        )

        agreements = sum([fund_agrees, risk_agrees, sent_agrees])
        if agreements == 3:
            confidence_adj = +10
        elif agreements == 2:
            confidence_adj = +5
        elif agreements == 1:
            confidence_adj = -5
        else:
            confidence_adj = -10

        if incomplete_data:
            confidence_adj -= 15

        confidence = max(30, min(95, base_confidence + confidence_adj))

        price_data = analyses["price_data"]

        sources = [
            price_data.get("realtime_quote", {}).get("current"),
            price_data.get("price_history", {}).get("summary", {}).get("latest_close"),
            price_data.get("price_history", {}).get("history", [{}])[-1].get("close")
                if price_data.get("price_history", {}).get("history") else None,
        ]
        current_price = next((float(v) for v in sources if v and 0 < float(v) < 100000), None)

        if not current_price:
            raise ValueError("Could not determine current price from price_data.json")

        self.logger.info(f"Current price: ${current_price}")

        if recommendation == "BUY":
            expected_return = 0.22 if confidence >= 70 else 0.15
        elif recommendation == "HOLD":
            expected_return = 0.07 if confidence >= 70 else 0.03
        else:
            expected_return = -0.18 if confidence >= 70 else -0.10

        price_target = round(current_price * (1 + expected_return), 2)
        upside_pct = round(expected_return * 100, 1)

        self.logger.info(f"Composite Score: {composite:.3f}")
        self.logger.info(
            f"Components: F={fundamental_component:.3f} R={risk_component:.3f} S={sentiment_component:.3f}"
        )
        self.logger.info(f"Recommendation: {recommendation}")
        self.logger.info(f"Confidence: {confidence}%")
        self.logger.info(f"Price Target: ${price_target} ({upside_pct:+.1f}%)")

        return {
            "composite_score": composite,
            "fundamental_component": fundamental_component,
            "risk_component": risk_component,
            "sentiment_component": sentiment_component,
            "recommendation": recommendation,
            "confidence": confidence,
            "current_price": current_price,
            "price_target": price_target,
            "upside_downside_pct": upside_pct,
            "agreements": agreements,
            "fundamental_score": fundamental_score,
            "risk_score": risk_score,
            "sentiment_score": sentiment_score,
        }

    def resolve_disagreements(self, analyses: dict, scores: dict) -> dict:
        """LLM CALL 1 — Disagreement resolution."""
        self.logger.info("Pass 1/3: Resolving analyst disagreements...")

        fundamental_verdict = analyses["fundamental"].get("verdict", "UNKNOWN")
        strengths = analyses["fundamental"].get("key_strengths", [])[:3]
        weaknesses = analyses["fundamental"].get("key_weaknesses", [])[:3]
        risk_verdict = analyses["risk"].get("risk_verdict", "UNKNOWN") if analyses["risk"] else "UNKNOWN"
        top_risks = [r["name"] for r in analyses["risk"].get("top_3_risks", [])] if analyses["risk"] else []
        bear_title = analyses["risk"].get("bear_case_title", "N/A") if analyses["risk"] else "N/A"
        sent_verdict = analyses["sentiment"].get("sentiment_verdict", "UNKNOWN") if analyses["sentiment"] else "UNKNOWN"
        momentum = analyses["sentiment"].get("momentum", "unknown") if analyses["sentiment"] else "unknown"
        key_event = analyses["sentiment"].get("most_important_event", "N/A") if analyses["sentiment"] else "N/A"

        prompt = DISAGREEMENT_RESOLUTION_PROMPT.format(
            fundamental_verdict=fundamental_verdict,
            fundamental_score=scores["fundamental_score"],
            strengths=", ".join(strengths),
            weaknesses=", ".join(weaknesses),
            risk_verdict=risk_verdict,
            risk_score=scores["risk_score"],
            top_risks=", ".join(top_risks),
            bear_case_title=bear_title,
            sentiment_verdict=sent_verdict,
            sentiment_score=scores["sentiment_score"],
            momentum=momentum,
            key_event=key_event,
            composite_score=scores["composite_score"],
            recommendation=scores["recommendation"],
        )

        result = self.llm_call(LEAD_ANALYST_SYSTEM_PROMPT, prompt)
        if not result:
            result = {
                "verdicts_summary": {
                    "fundamental": fundamental_verdict,
                    "risk": risk_verdict,
                    "sentiment": sent_verdict,
                    "agreement_level": "partial",
                },
                "conflict_resolution": "Quantitative composite score used as primary decision framework.",
                "dominant_factor": "fundamental",
                "dominant_factor_reasoning": "Fundamental quality is the most durable signal.",
                "key_swing_factor": "Risk score normalization",
                "investment_horizon": "medium_term (6-18m)",
            }
        return result

    def write_bull_bear_cases(self, analyses: dict, scores: dict, resolution: dict) -> dict:
        """LLM CALL 2 — Bull and bear case writing."""
        self.logger.info("Pass 2/3: Writing bull and bear cases...")

        company_name = analyses["profile"].get("name", self.ticker)
        strengths = analyses["fundamental"].get("key_strengths", [])
        weaknesses = analyses["fundamental"].get("key_weaknesses", [])
        thesis = analyses["fundamental"].get("investment_thesis", "")
        top_risks = analyses["risk"].get("top_3_risks", []) if analyses["risk"] else []
        bear_title = analyses["risk"].get("bear_case_title", "") if analyses["risk"] else ""
        downside = analyses["risk"].get("downside_price_target", "") if analyses["risk"] else ""
        sent_verdict = analyses["sentiment"].get("sentiment_verdict", "") if analyses["sentiment"] else ""
        bull_signals = analyses["sentiment"].get("key_bullish_signals", []) if analyses["sentiment"] else []
        bear_signals = analyses["sentiment"].get("key_bearish_signals", []) if analyses["sentiment"] else []
        catalysts = analyses["sentiment"].get("anticipated_catalysts", []) if analyses["sentiment"] else []

        prompt = BULL_BEAR_CASE_PROMPT.format(
            company_name=company_name,
            ticker=self.ticker,
            current_price=scores["current_price"],
            recommendation=scores["recommendation"],
            composite_score=scores["composite_score"],
            strengths=", ".join(strengths),
            weaknesses=", ".join(weaknesses),
            investment_thesis=thesis,
            risk_score=scores["risk_score"],
            top_risks=", ".join(r["name"] for r in top_risks),
            bear_case_title=bear_title,
            downside_target=downside,
            sentiment_verdict=sent_verdict,
            sentiment_score=scores["sentiment_score"],
            bullish_signals=", ".join(bull_signals),
            bearish_signals=", ".join(bear_signals),
            catalysts=", ".join(catalysts),
        )

        result = self.llm_call(LEAD_ANALYST_SYSTEM_PROMPT, prompt)
        if not result:
            result = {
                "bull_case": {
                    "title": "Strong fundamentals support upside",
                    "thesis": "The company's high-quality fundamentals and positive sentiment create a constructive setup.",
                    "key_drivers": [
                        {"driver": "Business Quality", "detail": "Strong moat and recurring revenue"},
                        {"driver": "Sentiment Momentum", "detail": "Accelerating positive coverage"},
                        {"driver": "Capital Allocation", "detail": "Significant FCF generation"},
                    ],
                    "upside_scenario": "Continued earnings growth and multiple expansion",
                    "price_target_bull": str(round(scores["current_price"] * 1.30, 2)),
                },
                "bear_case": {
                    "title": "High risk score limits upside",
                    "thesis": "Elevated risk score signals vulnerability to macro and sector headwinds.",
                    "key_risks": [
                        {"risk": "Risk Score", "detail": f"{scores['risk_score']}/10 risk rating"},
                        {"risk": "Valuation", "detail": "Premium multiple leaves little margin for error"},
                        {"risk": "Macro", "detail": "Rate environment and recession risk"},
                    ],
                    "downside_scenario": "Risk materialization causes multiple contraction",
                    "price_target_bear": str(downside or round(scores["current_price"] * 0.80, 2)),
                },
            }
        return result

    def write_investment_memo(
        self, analyses: dict, scores: dict, resolution: dict, bull_bear: dict
    ) -> dict:
        """LLM CALL 3 — Full investment memo writing."""
        self.logger.info("Pass 3/3: Writing investment memo...")

        company_name = analyses["profile"].get("name", self.ticker)
        sector = (
            analyses["profile"].get("finnhubIndustry") or
            analyses["profile"].get("sector") or
            analyses["profile"].get("industry") or
            "Technology"
        )
        industry = (
            analyses["profile"].get("industry") or
            analyses["profile"].get("finnhubIndustry") or
            "Consumer Electronics"
        )
        market_cap = analyses["profile"].get("market_cap", 0) or 0
        market_cap_b = market_cap / 1_000_000_000 if market_cap else 0

        top_risks = analyses["risk"].get("top_3_risks", []) if analyses["risk"] else []
        catalysts = analyses["sentiment"].get("anticipated_catalysts", []) if analyses["sentiment"] else []
        key_drivers = bull_bear.get("bull_case", {}).get("key_drivers", [])

        key_drivers_text = "\n".join(
            f"{i+1}. {d.get('driver', '')}: {d.get('detail', '')}"
            for i, d in enumerate(key_drivers)
        )
        top_risks_text = "\n".join(
            f"{i+1}. {r.get('name', r) if isinstance(r, dict) else r}"
            for i, r in enumerate(top_risks)
        )
        catalysts_text = "\n".join(f"- {c}" for c in catalysts)

        fundamental_verdict = analyses["fundamental"].get("verdict", "")
        risk_rating = analyses["risk"].get("risk_rating", "") if analyses["risk"] else ""
        sentiment_verdict = analyses["sentiment"].get("sentiment_verdict", "") if analyses["sentiment"] else ""
        key_event = analyses["sentiment"].get("most_important_event", "N/A") if analyses["sentiment"] else "N/A"
        conflict_resolution = resolution.get("conflict_resolution", "")
        investment_thesis = analyses["fundamental"].get("investment_thesis", "")
        bear_title = bull_bear.get("bear_case", {}).get("title", "")
        bear_thesis = bull_bear.get("bear_case", {}).get("thesis", "")
        bull_title = bull_bear.get("bull_case", {}).get("title", "")
        bull_thesis = bull_bear.get("bull_case", {}).get("thesis", "")

        prompt = INVESTMENT_MEMO_PROMPT.format(
            company_name=company_name,
            ticker=self.ticker,
            sector=sector,
            industry=industry,
            current_price=scores["current_price"],
            market_cap_b=market_cap_b,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            recommendation=scores["recommendation"],
            price_target=scores["price_target"],
            confidence=scores["confidence"],
            composite_score=scores["composite_score"],
            fundamental_score=scores["fundamental_score"],
            fundamental_verdict=fundamental_verdict,
            investment_thesis=investment_thesis,
            risk_score=scores["risk_score"],
            risk_rating=risk_rating,
            conflict_resolution=conflict_resolution,
            sentiment_score=scores["sentiment_score"],
            sentiment_verdict=sentiment_verdict,
            key_event=key_event,
            bull_title=bull_title,
            bull_thesis=bull_thesis,
            bear_title=bear_title,
            bear_thesis=bear_thesis,
            key_drivers_text=key_drivers_text,
            top_risks_text=top_risks_text,
            catalysts_text=catalysts_text,
        )

        result = self.llm_call(LEAD_ANALYST_SYSTEM_PROMPT, prompt)
        if not result:
            result = {
                "executive_summary": f"{company_name} ({self.ticker}) receives a {scores['recommendation']} recommendation with {scores['confidence']}% confidence. The composite score of {scores['composite_score']:.2f} reflects strong fundamentals offset by elevated risk. Our 12-month price target is ${scores['price_target']}.",
                "business_overview": f"{company_name} is a leading company in the {industry} sector.",
                "investment_thesis": investment_thesis,
                "key_financial_metrics": {
                    "current_price": scores["current_price"],
                    "price_target_12m": scores["price_target"],
                    "upside_downside_pct": scores["upside_downside_pct"],
                    "pe_ratio": 0.0,
                    "net_margin_pct": 0.0,
                    "fcf_billions": 0.0,
                    "revenue_growth_yoy_pct": 0.0,
                },
                "key_drivers": [d.get("driver", "") for d in key_drivers],
                "key_risks": [r.get("name", "") if isinstance(r, dict) else r for r in top_risks],
                "catalysts": catalysts[:3],
                "monitoring_checklist": [
                    "Quarterly earnings vs consensus estimates",
                    "Risk factor developments",
                    "Sentiment momentum trend",
                    "Price target revision by sell-side analysts",
                ],
                "analyst_notes": "AI-generated research. Not financial advice.",
            }

        # Overwrite key_financial_metrics with actual computed values
        result.setdefault("key_financial_metrics", {})
        result["key_financial_metrics"]["current_price"] = scores["current_price"]
        result["key_financial_metrics"]["price_target_12m"] = scores["price_target"]
        result["key_financial_metrics"]["upside_downside_pct"] = scores["upside_downside_pct"]

        # Fill from fundamental financial_health if available
        fh = analyses["fundamental"].get("financial_health", {})
        prof = fh.get("profitability", {})
        cf = fh.get("cash_flow", {})
        if prof.get("net_margin"):
            result["key_financial_metrics"]["net_margin_pct"] = float(prof["net_margin"])
        if cf.get("free_cash_flow"):
            result["key_financial_metrics"]["fcf_billions"] = float(cf["free_cash_flow"])

        # PE and revenue growth from key_metrics
        bfin = analyses["metrics"].get("basic_financials", {})
        pe = bfin.get("valuation", {}).get("pe_ttm")
        if pe:
            result["key_financial_metrics"]["pe_ratio"] = round(float(pe), 1)
        rev_growth = bfin.get("growth", {}).get("revenue_growth_ttm")
        if rev_growth:
            result["key_financial_metrics"]["revenue_growth_yoy_pct"] = round(float(rev_growth), 1)

        return result

    def generate_markdown_memo(
        self,
        analyses: dict,
        scores: dict,
        resolution: dict,
        bull_bear: dict,
        memo: dict,
    ) -> str:
        """Generate the human-readable markdown investment memo. Pure Python — no LLM."""
        company_name = analyses["profile"].get("name", self.ticker)
        sector = (
            analyses["profile"].get("finnhubIndustry") or
            analyses["profile"].get("sector") or
            analyses["profile"].get("industry") or
            "Technology"
        )
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        market_cap = analyses["profile"].get("market_cap", 0) or 0
        market_cap_b = round(market_cap / 1_000_000_000, 1) if market_cap else 0

        fundamental_verdict = analyses["fundamental"].get("verdict", "")
        risk_verdict = analyses["risk"].get("risk_verdict", "") if analyses["risk"] else ""
        sentiment_verdict = analyses["sentiment"].get("sentiment_verdict", "") if analyses["sentiment"] else ""

        bull = bull_bear.get("bull_case", {})
        bear = bull_bear.get("bear_case", {})

        kfm = memo.get("key_financial_metrics", {})
        pe_ratio = kfm.get("pe_ratio", "N/A")
        net_margin = kfm.get("net_margin_pct", "N/A")
        fcf = kfm.get("fcf_billions", "N/A")
        rev_growth = kfm.get("revenue_growth_yoy_pct", "N/A")

        agreement_level = resolution.get("verdicts_summary", {}).get("agreement_level", "")
        conflict_resolution = resolution.get("conflict_resolution", "")
        dominant_factor = resolution.get("dominant_factor", "")
        dominant_factor_reasoning = resolution.get("dominant_factor_reasoning", "")
        key_swing_factor = resolution.get("key_swing_factor", "")

        catalysts = memo.get("catalysts", [])
        monitoring = memo.get("monitoring_checklist", [])
        key_drivers = memo.get("key_drivers", [])
        key_risks = memo.get("key_risks", [])

        bull_drivers_md = "\n".join(
            f"- **{d.get('driver', '')}**: {d.get('detail', '')}"
            for d in bull.get("key_drivers", [])
        )
        bear_risks_md = "\n".join(
            f"- **{r.get('risk', '')}**: {r.get('detail', '')}"
            for r in bear.get("key_risks", [])
        )
        catalysts_md = "\n".join(f"- {c}" for c in catalysts)
        monitoring_md = "\n".join(f"- [ ] {item}" for item in monitoring)

        rec = scores["recommendation"]
        star = "★" if rec == "BUY" else ("◆" if rec == "HOLD" else "▼")

        md = f"""# EQUITY RESEARCH — {self.ticker}
## {company_name} | {sector}
**{date}** | Analyst: AI Research Pipeline v1.0

---

## RECOMMENDATION: {rec} {star}

| Field | Value |
|-------|-------|
| Current Price | ${scores['current_price']} |
| 12-Month Price Target | ${scores['price_target']} |
| Expected Return | {scores['upside_downside_pct']:+.1f}% |
| Confidence | {scores['confidence']}% |
| Composite Score | {scores['composite_score']:.2f} / 1.00 |

---

## EXECUTIVE SUMMARY
{memo.get('executive_summary', '')}

---

## SCORECARD

| Analyst | Score | Verdict |
|---------|-------|---------|
| Fundamental | {scores['fundamental_score']}/10 | {fundamental_verdict} |
| Risk | {scores['risk_score']}/10 (RISK) | {risk_verdict} |
| Sentiment | {scores['sentiment_score']:+.2f} | {sentiment_verdict} |
| **COMPOSITE** | **{scores['composite_score']:.2f}** | **{rec}** |

---

## BUSINESS OVERVIEW
{memo.get('business_overview', '')}

---

## INVESTMENT THESIS
{memo.get('investment_thesis', '')}

---

## KEY FINANCIAL METRICS

| Metric | Value |
|--------|-------|
| P/E Ratio | {pe_ratio}x |
| Net Margin | {net_margin}% |
| Free Cash Flow | ${fcf}B |
| Revenue Growth YoY | {rev_growth}% |
| Market Cap | ${market_cap_b}B |

---

## BULL CASE — "{bull.get('title', '')}"
{bull.get('thesis', '')}

**Key Drivers:**
{bull_drivers_md}

**Upside Scenario:** {bull.get('upside_scenario', '')}
**Bull Price Target:** ${bull.get('price_target_bull', '')}

---

## BEAR CASE — "{bear.get('title', '')}"
{bear.get('thesis', '')}

**Key Risks:**
{bear_risks_md}

**Downside Scenario:** {bear.get('downside_scenario', '')}
**Bear Price Target:** ${bear.get('price_target_bear', '')}

---

## ANALYST DISAGREEMENTS & RESOLUTION
**Agreement Level:** {agreement_level}
{conflict_resolution}

**Dominant Factor:** {dominant_factor} — {dominant_factor_reasoning}

**Key Swing Factor:** {key_swing_factor}

---

## CATALYSTS TO WATCH
{catalysts_md}

---

## MONITORING CHECKLIST
{monitoring_md}

---

## ANALYST NOTES
{memo.get('analyst_notes', '')}

---
*Generated by AI Financial Analysis Pipeline | {date}*
*This is AI-generated research for educational purposes only.*
*Not financial advice. Always conduct your own due diligence.*
"""
        return md

    def print_summary(
        self,
        scores: dict,
        resolution: dict,
        bull_bear: dict,
        memo: dict,
        analyses: dict,
    ):
        """Print the Rich terminal final summary."""
        rec = scores["recommendation"]
        color = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(rec, "white")
        company_name = analyses["profile"].get("name", self.ticker)
        sector = (
            analyses["profile"].get("finnhubIndustry") or
            analyses["profile"].get("sector") or
            analyses["profile"].get("industry") or
            ""
        )
        exchange = analyses["profile"].get("exchange", "")
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")

        console.print()
        console.rule(
            f"[bold]INVESTMENT RECOMMENDATION — {self.ticker}[/bold]", style="dim"
        )
        console.print(f"  {company_name} | {sector} | {exchange}")
        console.print(f"  Generated: {date}")
        console.rule(style="dim")

        stars = "⭐  " if rec == "BUY" else ("◆  " if rec == "HOLD" else "▼  ")
        rec_text = (
            f"[bold {color}]{stars}{rec}{stars}[/bold {color}]\n"
            f"  Confidence: {scores['confidence']}%\n"
            f"  12M Price Target: ${scores['price_target']}\n"
            f"  Expected Return: {scores['upside_downside_pct']:+.1f}%"
        )
        console.print(Panel(rec_text, box=box.DOUBLE, border_style=color, padding=(1, 4)))

        console.print()
        console.print("[bold]COMPOSITE SCORE BREAKDOWN[/bold]")
        f_comp = scores["fundamental_component"]
        r_comp = scores["risk_component"]
        s_comp = scores["sentiment_component"]
        console.print(
            f"  Fundamental (40%):  {scores['fundamental_score']}/10  →  {f_comp:.3f}  ×0.40 = {f_comp*0.40:.3f}"
        )
        console.print(
            f"  Risk (35%):         {scores['risk_score']}/10  →  {r_comp:.3f}  ×0.35 = {r_comp*0.35:.3f}  ← inverted"
        )
        console.print(
            f"  Sentiment (25%):    {scores['sentiment_score']:+.2f}   →  {s_comp:.3f}  ×0.25 = {s_comp*0.25:.3f}"
        )
        console.print("  " + "─" * 55)
        console.print(
            f"  [bold]COMPOSITE SCORE:                        {scores['composite_score']:.3f}  ({rec})[/bold]"
        )

        console.print()
        fundamental_verdict = analyses["fundamental"].get("verdict", "")
        risk_verdict = analyses["risk"].get("risk_verdict", "") if analyses["risk"] else "N/A"
        sentiment_verdict = analyses["sentiment"].get("sentiment_verdict", "") if analyses["sentiment"] else "N/A"
        agreements = scores["agreements"]
        agreement_level = resolution.get("verdicts_summary", {}).get("agreement_level", "")
        console.print("[bold]ANALYST VERDICTS[/bold]")
        console.print(
            f"  Fundamental: {fundamental_verdict}    Risk: {risk_verdict}    Sentiment: {sentiment_verdict}"
        )
        console.print(f"  Agreement: {agreement_level} ({agreements}/3 agree)")

        console.print()
        bull_title = bull_bear.get("bull_case", {}).get("title", "")
        bear_title = bull_bear.get("bear_case", {}).get("title", "")
        console.print(f'[green]BULL CASE:[/green] "{bull_title}"')
        console.print(f'[red]BEAR CASE:[/red] "{bear_title}"')

        top_risks = analyses["risk"].get("top_3_risks", []) if analyses["risk"] else []
        if top_risks:
            console.print()
            console.print("[bold]TOP 3 RISKS[/bold]")
            for r in top_risks[:3]:
                console.print(f"  #{r.get('rank', '')} {r.get('name', '')}")

        catalysts = memo.get("catalysts", [])
        if catalysts:
            console.print()
            console.print("[bold]KEY CATALYSTS[/bold]")
            for c in catalysts[:3]:
                console.print(f"  → {c}")

        base = f"{self.config.DATA_DIR}/{self.ticker}/analysis"
        console.print()
        console.print("[bold]FILES SAVED:[/bold]")
        console.print(f"  📄 {base}/investment_memo.json")
        console.print(f"  📝 {base}/investment_memo.md")
        console.print()

    def analyze(self, refresh: bool = False) -> dict:
        """Main method — orchestrates the final synthesis."""
        cached = self.load_result("investment_memo.json")
        if cached and not refresh:
            meta = cached.get("_metadata", {})
            generated_at = meta.get("generated_at", "")
            if generated_at:
                from datetime import timedelta
                generated_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - generated_dt
                if age.total_seconds() < 86400:
                    console.print(
                        f"[dim]Loading cached investment memo for {self.ticker} "
                        f"(generated {int(age.total_seconds()//3600)}h ago)[/dim]"
                    )
                    return cached

        console.print(f"\n[bold]🎯 Running Lead Analyst Synthesis for {self.ticker}...[/bold]")

        # Step 2: Load analyses
        analyses = self.load_all_analyses()

        # Step 3: Show which analyses are available
        fa = analyses["fundamental"]
        ra = analyses["risk"]
        sa = analyses["sentiment"]
        console.print(
            f"  ✅ Fundamental Analysis (score: {fa.get('overall_score', 'N/A')}/10)"
        )
        if ra:
            console.print(
                f"  ✅ Risk Analysis (risk: {ra.get('overall_risk_score', 'N/A')}/10)"
            )
        else:
            console.print("  ⚠️  Risk Analysis missing — using defaults")
        if sa:
            console.print(
                f"  ✅ Sentiment Analysis (score: {sa.get('weighted_sentiment_score', 'N/A'):+})"
            )
        else:
            console.print("  ⚠️  Sentiment Analysis missing — using defaults")

        # Step 4: Compute composite score (pure math)
        scores = self.compute_composite_score(analyses)

        # Step 5: Resolve disagreements (LLM call 1)
        resolution = self.resolve_disagreements(analyses, scores)

        # Step 6: Write bull/bear cases (LLM call 2)
        bull_bear = self.write_bull_bear_cases(analyses, scores, resolution)

        # Step 7: Write investment memo (LLM call 3)
        memo = self.write_investment_memo(analyses, scores, resolution, bull_bear)

        # Step 8: Generate markdown memo (pure Python)
        markdown = self.generate_markdown_memo(analyses, scores, resolution, bull_bear, memo)

        company_name = analyses["profile"].get("name", self.ticker)
        fundamental_verdict = fa.get("verdict")
        risk_verdict = ra.get("risk_verdict") if ra else None
        sentiment_verdict = sa.get("sentiment_verdict") if sa else None

        # Step 9: Assemble final result
        result = {
            "ticker": self.ticker,
            "company_name": company_name,
            "analysis_type": "investment_memo",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recommendation": scores["recommendation"],
            "confidence": scores["confidence"],
            "composite_score": scores["composite_score"],
            "current_price": scores["current_price"],
            "price_target_12m": scores["price_target"],
            "expected_return_pct": scores["upside_downside_pct"],
            "score_breakdown": {
                "fundamental_score": scores["fundamental_score"],
                "fundamental_component": scores["fundamental_component"],
                "risk_score": scores["risk_score"],
                "risk_component": scores["risk_component"],
                "sentiment_score": scores["sentiment_score"],
                "sentiment_component": scores["sentiment_component"],
            },
            "verdicts": {
                "fundamental": fundamental_verdict,
                "risk": risk_verdict,
                "sentiment": sentiment_verdict,
                "agreement_level": resolution.get("verdicts_summary", {}).get("agreement_level"),
            },
            "resolution": resolution,
            "bull_case": bull_bear.get("bull_case", {}),
            "bear_case": bull_bear.get("bear_case", {}),
            "memo": memo,
            "executive_summary": memo.get("executive_summary", ""),
            "investment_thesis": memo.get("investment_thesis", ""),
            "key_drivers": memo.get("key_drivers", []),
            "key_risks": memo.get("key_risks", []),
            "catalysts": memo.get("catalysts", []),
            "monitoring_checklist": memo.get("monitoring_checklist", []),
        }

        # Step 10: Save JSON
        self.save_result(result, "investment_memo.json")

        # Step 11: Save markdown
        md_path = f"{self.config.DATA_DIR}/{self.ticker}/analysis/investment_memo.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        self.logger.info(f"Saved: {md_path}")

        # Step 12: Print rich summary
        self.print_summary(scores, resolution, bull_bear, memo, analyses)

        return result
