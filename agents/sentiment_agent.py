from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from tqdm import tqdm

from agents.base_agent import BaseAgent
from agents.prompts.sentiment_prompts import (
    SENTIMENT_SYSTEM_PROMPT,
    ARTICLE_BATCH_SENTIMENT_PROMPT,
    THEME_ANALYSIS_PROMPT,
    SENTIMENT_SYNTHESIS_PROMPT,
)

console = Console()

_CACHE_TTL_HOURS = 24
_BATCH_SIZE = 10
_MAX_ARTICLE_CHARS = 300
_MAX_CONTEXT_BATCH_CHARS = 5500


class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent.
    Processes news articles in batches, classifies each, computes weighted
    sentiment scores with recency decay, identifies dominant themes, and
    produces a market mood verdict.
    """

    SOURCE_WEIGHTS: dict[str, float] = {
        "bloomberg": 1.0,
        "reuters": 1.0,
        "financial times": 1.0,
        "wall street journal": 1.0,
        "wsj": 1.0,
        "barron's": 0.9,
        "ft": 0.9,
        "cnbc": 0.8,
        "fortune": 0.8,
        "marketwatch": 0.8,
        "business insider": 0.7,
        "seeking alpha": 0.7,
        "motley fool": 0.6,
        "yahoo finance": 0.6,
        "investopedia": 0.5,
        "techcrunch": 0.5,
        "the verge": 0.4,
        "default": 0.5,
    }

    # ── Data loaders ──────────────────────────────────────────────────────────

    def load_articles(self) -> list[dict]:
        path = f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"News articles not found. Run Step 1 first:\n"
                f"  python main.py {self.ticker}"
            )
        with open(path) as f:
            data = json.load(f)
        return data.get("articles", [])

    def load_prior_analyses(self) -> tuple[dict, dict]:
        fundamental: dict = {}
        risk: dict = {}

        fund_result = self.load_result("fundamental_analysis.json")
        if fund_result:
            fundamental = fund_result
        else:
            self.logger.warning("fundamental_analysis.json not found — running without it.")

        risk_result = self.load_result("risk_analysis.json")
        if risk_result:
            risk = risk_result
        else:
            self.logger.warning("risk_analysis.json not found — running without it.")

        return fundamental, risk

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess_articles(self, articles: list[dict]) -> list[dict]:
        now = datetime.now(timezone.utc)
        enriched: list[dict] = []

        for article in articles:
            headline = article.get("headline") or ""
            published_at = article.get("published_at") or ""
            if not headline.strip() or not published_at:
                continue

            # Parse date
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except ValueError:
                continue

            age_days = (now - dt).days
            recency_weight = math.exp(-0.05 * age_days)

            # Source weight — check against known sources (partial match)
            source_raw = (article.get("source") or "").lower()
            source_weight = self.SOURCE_WEIGHTS["default"]
            for key, weight in self.SOURCE_WEIGHTS.items():
                if key != "default" and key in source_raw:
                    source_weight = weight
                    break

            combined_weight = recency_weight * source_weight

            summary = article.get("summary") or ""
            article_text = (
                f"[{published_at[:10]}] {article.get('source', '')}: "
                f"{headline}. {summary}"
            )[:_MAX_ARTICLE_CHARS]

            enriched.append({
                **article,
                "parsed_date": dt,
                "age_days": age_days,
                "recency_weight": recency_weight,
                "source_weight": source_weight,
                "combined_weight": combined_weight,
                "article_text": article_text,
            })

        enriched.sort(key=lambda a: a["parsed_date"], reverse=True)

        if enriched:
            newest = enriched[0]["published_at"][:10]
            oldest = enriched[-1]["published_at"][:10]
            self.logger.info(
                f"Preprocessed {len(enriched)} articles, date range: {oldest} to {newest}"
            )

        return enriched

    # ── Pass 1: Batch classification ──────────────────────────────────────────

    def classify_articles_in_batches(self, articles: list[dict]) -> list[dict]:
        # Load company name for prompts
        news_path = f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json"
        company_name = self.ticker
        if os.path.exists(news_path):
            with open(news_path) as f:
                company_name = json.load(f).get("company_name", self.ticker)

        batches = [articles[i:i + _BATCH_SIZE] for i in range(0, len(articles), _BATCH_SIZE)]
        classified = list(articles)  # copy

        with tqdm(total=len(articles), desc="Classifying articles", unit="art") as pbar:
            for batch_idx, batch in enumerate(batches):
                # Format articles text, staying under context limit
                lines: list[str] = []
                for local_idx, article in enumerate(batch):
                    lines.append(f"Article {local_idx}: {article['article_text']}")
                articles_text = "\n".join(lines)

                # Keep batch prompt under char limit
                if len(articles_text) > _MAX_CONTEXT_BATCH_CHARS:
                    articles_text = articles_text[:_MAX_CONTEXT_BATCH_CHARS] + "\n[truncated]"

                try:
                    response = self.llm_call(
                        system_prompt=SENTIMENT_SYSTEM_PROMPT,
                        user_message=ARTICLE_BATCH_SENTIMENT_PROMPT.format(
                            ticker=self.ticker,
                            company_name=company_name,
                            articles_text=articles_text,
                        ),
                    )
                    llm_articles = response.get("articles", []) if isinstance(response, dict) else []
                except Exception as exc:
                    self.logger.warning(f"Batch {batch_idx} failed: {exc} — defaulting to neutral")
                    llm_articles = []

                # Map results back by article_index
                result_map: dict[int, dict] = {}
                for entry in llm_articles:
                    idx = entry.get("article_index")
                    if isinstance(idx, int) and 0 <= idx < len(batch):
                        result_map[idx] = entry

                global_offset = batch_idx * _BATCH_SIZE
                for local_idx, article in enumerate(batch):
                    global_idx = global_offset + local_idx
                    entry = result_map.get(local_idx, {})
                    classified[global_idx] = {
                        **article,
                        "sentiment_label": entry.get("sentiment", "neutral"),
                        "sentiment_confidence": float(entry.get("confidence", 0.5)),
                        "key_point": entry.get("key_point", ""),
                        "market_impact": entry.get("market_impact", "uncertain"),
                        "detected_themes": entry.get("themes", []),
                    }

                pbar.update(len(batch))

        counts = Counter(a["sentiment_label"] for a in classified)
        self.logger.info(
            f"Classification complete: {counts.get('bullish', 0)} bullish, "
            f"{counts.get('bearish', 0)} bearish, {counts.get('neutral', 0)} neutral"
        )
        return classified

    # ── Score computation ──────────────────────────────────────────────────────

    def compute_sentiment_scores(self, articles: list[dict]) -> dict:
        now = datetime.now(timezone.utc)
        cutoff_7d = now - timedelta(days=7)

        contributions: list[float] = []
        weights: list[float] = []
        recent_contrib: list[float] = []
        recent_weights: list[float] = []
        older_contrib: list[float] = []
        older_weights: list[float] = []

        bullish = bearish = neutral = 0
        hi_bullish = hi_bearish = 0

        for article in articles:
            label = article.get("sentiment_label", "neutral")
            conf = float(article.get("sentiment_confidence", 0.5))
            w = float(article.get("combined_weight", 0.5))
            dt = article.get("parsed_date")

            if label == "bullish":
                numeric = conf
                bullish += 1
                if conf >= 0.8:
                    hi_bullish += 1
            elif label == "bearish":
                numeric = -conf
                bearish += 1
                if conf >= 0.8:
                    hi_bearish += 1
            else:
                numeric = 0.0
                neutral += 1

            contributions.append(numeric * w)
            weights.append(w)

            if dt and dt >= cutoff_7d:
                recent_contrib.append(numeric * w)
                recent_weights.append(w)
            else:
                older_contrib.append(numeric * w)
                older_weights.append(w)

        total = len(articles)

        def _safe_avg(c: list, w: list) -> float:
            total_w = sum(w)
            return max(-1.0, min(1.0, sum(c) / total_w)) if total_w > 0 else 0.0

        weighted_score = _safe_avg(contributions, weights)
        simple_score = _safe_avg(
            [c / w if w else 0 for c, w in zip(contributions, weights)],
            [1.0] * len(weights),
        )
        recent_score = _safe_avg(recent_contrib, recent_weights)
        older_score = _safe_avg(older_contrib, older_weights)
        momentum_delta = recent_score - older_score

        return {
            "weighted_score": round(weighted_score, 3),
            "simple_score": round(simple_score, 3),
            "recent_7d_score": round(recent_score, 3),
            "older_score": round(older_score, 3),
            "momentum_delta": round(momentum_delta, 3),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "bullish_pct": round(bullish / total * 100, 1) if total else 0.0,
            "bearish_pct": round(bearish / total * 100, 1) if total else 0.0,
            "neutral_pct": round(neutral / total * 100, 1) if total else 0.0,
            "total_articles": total,
            "high_confidence_bullish": hi_bullish,
            "high_confidence_bearish": hi_bearish,
        }

    # ── Pass 2: Theme analysis ────────────────────────────────────────────────

    def analyze_themes(
        self,
        articles: list[dict],
        scores: dict,
        fundamental: dict,
        risk: dict,
    ) -> dict:
        self.logger.info("Pass 2/3: Theme & Narrative Analysis...")
        console.print("[bold yellow]  Pass 2/3:[/bold yellow] Theme & Narrative Analysis...")

        # Load company name
        news_path = f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json"
        company_name = self.ticker
        if os.path.exists(news_path):
            with open(news_path) as f:
                company_name = json.load(f).get("company_name", self.ticker)

        # Classified summary (max 50 articles, 120 chars per line)
        summary_lines: list[str] = []
        for article in articles[:50]:
            label = article.get("sentiment_label", "neutral")
            key_point = (article.get("key_point") or article.get("headline") or "")[:100]
            summary_lines.append(f"- [{label}] {key_point}")
        classified_summary = "\n".join(summary_lines)

        # Theme frequency counts
        all_themes: list[str] = []
        for article in articles:
            all_themes.extend(article.get("detected_themes") or [])
        theme_counter = Counter(all_themes)
        theme_counts = ", ".join(f"{t}: {c}" for t, c in theme_counter.most_common(12))

        # Recent headlines (last 7 days, max 10)
        now = datetime.now(timezone.utc)
        cutoff_7d = now - timedelta(days=7)
        recent_lines: list[str] = []
        for article in articles:
            dt = article.get("parsed_date")
            if dt and dt >= cutoff_7d:
                headline = (article.get("headline") or "")[:80]
                source = article.get("source", "")
                date = article.get("published_at", "")[:10]
                recent_lines.append(f"- {headline} [{source}, {date}]")
            if len(recent_lines) >= 10:
                break
        recent_headlines = "\n".join(recent_lines) or "No articles in last 7 days."

        # Top risks from risk analysis
        top_3 = risk.get("top_3_risks", [])
        if top_3:
            top_risks = "\n".join(
                f"{r.get('rank', i+1)}. {r.get('name', '')} — {r.get('one_line', '')}"
                for i, r in enumerate(top_3)
            )
        else:
            top_risks = "Risk analysis not available."

        result = self.llm_call(
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_message=THEME_ANALYSIS_PROMPT.format(
                ticker=self.ticker,
                company_name=company_name,
                classified_summary=classified_summary[:3000],
                theme_counts=theme_counts,
                recent_headlines=recent_headlines,
                top_risks=top_risks,
            ),
        )

        return result

    # ── Pass 3: Synthesis ─────────────────────────────────────────────────────

    def synthesize(
        self,
        articles: list[dict],
        scores: dict,
        themes: dict,
        fundamental: dict,
        risk: dict,
    ) -> dict:
        self.logger.info("Pass 3/3: Synthesizing sentiment verdict...")
        console.print("[bold yellow]  Pass 3/3:[/bold yellow] Synthesizing sentiment verdict...")

        news_path = f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json"
        company_name = self.ticker
        lookback_days = self.config.NEWS_LOOKBACK_DAYS
        if os.path.exists(news_path):
            with open(news_path) as f:
                raw = json.load(f)
                company_name = raw.get("company_name", self.ticker)
                lookback_days = raw.get("lookback_days", lookback_days)

        fundamental_verdict = fundamental.get("verdict", "UNKNOWN")
        risk_verdict = risk.get("risk_verdict", "UNKNOWN")

        # Format dominant themes
        theme_lines: list[str] = []
        for i, t in enumerate(themes.get("dominant_themes", []), 1):
            theme_lines.append(
                f"{i}. {t.get('theme', '')} ({t.get('sentiment', '')}, {t.get('intensity', '')} intensity):\n"
                f"   {t.get('narrative', '')}"
            )
        dominant_themes_text = "\n".join(theme_lines) or "No dominant themes identified."

        concerns = themes.get("emerging_concerns", [])
        emerging_concerns_text = "\n".join(f"- {c}" for c in concerns) or "None identified."

        catalysts = themes.get("anticipated_catalysts", [])
        catalysts_text = "\n".join(f"- {c}" for c in catalysts) or "None identified."

        result = self.llm_call(
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_message=SENTIMENT_SYNTHESIS_PROMPT.format(
                ticker=self.ticker,
                company_name=company_name,
                lookback_days=lookback_days,
                total_articles=scores["total_articles"],
                bullish_count=scores["bullish_count"],
                bullish_pct=scores["bullish_pct"],
                bearish_count=scores["bearish_count"],
                bearish_pct=scores["bearish_pct"],
                neutral_count=scores["neutral_count"],
                neutral_pct=scores["neutral_pct"],
                weighted_score=scores["weighted_score"],
                dominant_themes_text=dominant_themes_text,
                emerging_concerns_text=emerging_concerns_text,
                catalysts_text=catalysts_text,
                fundamental_verdict=fundamental_verdict,
                risk_verdict=risk_verdict,
            ),
        )

        return result

    # ── Summary printer ───────────────────────────────────────────────────────

    def print_summary(self, result: dict):
        score = result.get("weighted_sentiment_score", 0.0)
        verdict = result.get("sentiment_verdict", "UNKNOWN")
        confidence = result.get("verdict_confidence", 0)
        momentum = result.get("momentum", "unknown")
        overall = result.get("overall_sentiment", "unknown")
        alignment = result.get("news_vs_fundamentals_alignment", "N/A")
        total = result.get("total_articles", 0)
        period = result.get("period_days", 30)
        generated_at = result.get("_metadata", {}).get("generated_at", "")
        citations = result.get("_metadata", {}).get("citations", [])

        breakdown = result.get("article_breakdown", {})
        bullish_n = breakdown.get("bullish", 0)
        bearish_n = breakdown.get("bearish", 0)
        neutral_n = breakdown.get("neutral", 0)
        bullish_pct = breakdown.get("bullish_pct", 0)
        bearish_pct = breakdown.get("bearish_pct", 0)
        neutral_pct = 100 - bullish_pct - bearish_pct

        # Score color
        if score >= 0.3:
            score_color, score_label = "bold green", "BULLISH"
        elif score >= 0.1:
            score_color, score_label = "green", "LEANING BULLISH"
        elif score >= -0.1:
            score_color, score_label = "yellow", "NEUTRAL"
        elif score >= -0.3:
            score_color, score_label = "red", "LEANING BEARISH"
        else:
            score_color, score_label = "bold red", "BEARISH"

        console.print(
            Panel(
                f"[bold]SENTIMENT ANALYSIS — {self.ticker}[/bold]\n"
                f"Generated: {generated_at[:19].replace('T', ' ')} UTC\n"
                f"Articles analyzed: {total} (last {period} days)",
                box=box.ROUNDED,
                style="bold white",
            )
        )

        console.print(
            f"\n[bold]SENTIMENT SCORE:[/bold]  "
            f"[{score_color}]{score:+.2f}  ({score_label})[/{score_color}]"
        )
        console.print(f"[bold]VERDICT:[/bold]          [bold]{verdict}[/bold]  (Confidence: {confidence}%)")
        console.print(f"[bold]MOMENTUM:[/bold]         {momentum}\n")

        # ASCII bar chart
        def _bar(count: int, total: int, width: int = 20) -> str:
            filled = round(count / total * width) if total else 0
            return "█" * filled + "░" * (width - filled)

        console.print("[bold]ARTICLE BREAKDOWN[/bold]")
        console.print(f"  [green]Bullish:[/green]  {bullish_n:3d} ({bullish_pct:.0f}%)  {_bar(bullish_n, total)}")
        console.print(f"  [yellow]Neutral:[/yellow]  {neutral_n:3d} ({neutral_pct:.0f}%)  {_bar(neutral_n, total)}")
        console.print(f"  [red]Bearish:[/red]  {bearish_n:3d} ({bearish_pct:.0f}%)  {_bar(bearish_n, total)}\n")

        # Top themes table
        themes_list = result.get("dominant_themes", [])
        if themes_list:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
            table.add_column("#", min_width=3)
            table.add_column("THEME", min_width=28)
            table.add_column("SENT", min_width=8)
            table.add_column("ARTICLES", min_width=8)
            table.add_column("INTENSITY", min_width=9)
            for i, t in enumerate(themes_list[:5], 1):
                sent = t.get("sentiment", "")
                sent_color = "green" if sent == "bullish" else ("red" if sent == "bearish" else "yellow")
                table.add_row(
                    str(i),
                    t.get("theme", ""),
                    f"[{sent_color}]{sent}[/{sent_color}]",
                    str(t.get("article_count", "")),
                    t.get("intensity", ""),
                )
            console.print("[bold]TOP THEMES[/bold]")
            console.print(table)

        # Key signals
        bullish_signals = result.get("key_bullish_signals", [])
        bearish_signals = result.get("key_bearish_signals", [])
        if bullish_signals or bearish_signals:
            console.print("[bold]KEY SIGNALS[/bold]")
            if bullish_signals:
                console.print(f"  [green]Bullish:[/green] {', '.join(bullish_signals[:3])}")
            if bearish_signals:
                console.print(f"  [red]Bearish:[/red] {', '.join(bearish_signals[:2])}")

        event = result.get("most_important_event", "")
        if event:
            console.print(f"\n[bold]MOST IMPORTANT EVENT:[/bold]\n  \"{event}\"")

        console.print(f"\n[bold]ALIGNMENT WITH FUNDAMENTALS:[/bold] {alignment}")

        watch_for = result.get("watch_for", [])
        if watch_for:
            console.print(f"[bold]WATCH FOR:[/bold] {', '.join(watch_for[:3])}")

        console.print(f"\n[dim]SOURCES CITED: {len(citations)}[/dim]\n")

    # ── Main orchestrator ─────────────────────────────────────────────────────

    def analyze(self, refresh: bool = False) -> dict:
        """Orchestrate all sentiment passes. Caches result for 24 hours."""
        if not refresh:
            cached = self.load_result("sentiment_analysis.json")
            if cached:
                meta = cached.get("_metadata", {})
                generated_at = meta.get("generated_at", "")
                if generated_at:
                    try:
                        from datetime import timedelta
                        age = datetime.now(timezone.utc) - datetime.fromisoformat(generated_at)
                        if age.total_seconds() < _CACHE_TTL_HOURS * 3600:
                            console.print(
                                "[dim]Using cached sentiment analysis "
                                "(run with --refresh to regenerate)[/dim]"
                            )
                            return cached
                    except ValueError:
                        pass

        console.print(f"\n[bold]Sentiment Analysis — {self.ticker}[/bold]")

        articles = self.load_articles()
        fundamental, risk = self.load_prior_analyses()

        # Load metadata for display
        news_path = f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json"
        lookback_days = self.config.NEWS_LOOKBACK_DAYS
        if os.path.exists(news_path):
            with open(news_path) as f:
                raw_news = json.load(f)
                lookback_days = raw_news.get("lookback_days", lookback_days)

        console.print(f"   Articles: {len(articles)} | Period: last {lookback_days} days\n")

        preprocessed = self.preprocess_articles(articles)
        classified = self.classify_articles_in_batches(preprocessed)
        scores = self.compute_sentiment_scores(classified)
        themes = self.analyze_themes(classified, scores, fundamental, risk)
        synthesis = self.synthesize(classified, scores, themes, fundamental, risk)

        # Determine momentum label from scores
        delta = scores.get("momentum_delta", 0.0)
        recent = scores.get("recent_7d_score", 0.0)
        if delta > 0.1:
            momentum_label = "accelerating_positive"
        elif delta < -0.1:
            momentum_label = "accelerating_negative"
        elif recent > 0.1:
            momentum_label = "stable_positive"
        elif recent < -0.1:
            momentum_label = "stable_negative"
        else:
            momentum_label = "neutral"

        # Prefer LLM momentum if provided
        momentum = synthesis.get("momentum") or momentum_label

        result = {
            "ticker": self.ticker,
            "analysis_type": "sentiment",
            "period_days": lookback_days,
            "total_articles": scores["total_articles"],
            "article_breakdown": {
                "bullish": scores["bullish_count"],
                "bearish": scores["bearish_count"],
                "neutral": scores["neutral_count"],
                "bullish_pct": scores["bullish_pct"],
                "bearish_pct": scores["bearish_pct"],
            },
            "weighted_sentiment_score": scores["weighted_score"],
            "recent_7d_score": scores["recent_7d_score"],
            "momentum": momentum,
            "overall_sentiment": synthesis.get("overall_sentiment"),
            "sentiment_verdict": synthesis.get("sentiment_verdict"),
            "verdict_confidence": synthesis.get("verdict_confidence"),
            "dominant_themes": themes.get("dominant_themes", []),
            "emerging_concerns": themes.get("emerging_concerns", []),
            "anticipated_catalysts": themes.get("anticipated_catalysts", []),
            "key_bullish_signals": synthesis.get("key_bullish_signals", []),
            "key_bearish_signals": synthesis.get("key_bearish_signals", []),
            "most_important_event": synthesis.get("most_important_recent_event"),
            "news_vs_fundamentals_alignment": synthesis.get("news_vs_fundamentals_alignment"),
            "watch_for": synthesis.get("watch_for", []),
            "classified_articles": classified,
        }

        self.save_result(result, "sentiment_analysis.json")
        self.print_summary(result)

        return result
