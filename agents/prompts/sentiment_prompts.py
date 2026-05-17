SENTIMENT_SYSTEM_PROMPT = """
You are a financial news analyst specializing in market sentiment analysis
and narrative tracking for equity research.

Your job is to read financial news articles and extract:
1. Sentiment signals (bullish/bearish/neutral)
2. Key market themes and narratives
3. How the news relates to known company risks and opportunities
4. Whether market sentiment is improving or deteriorating

STRICT RULES:
1. Base ALL analysis on the provided article text only.
2. Classify sentiment based on market implications, not tone.
   e.g. "Apple beats earnings by 5%" = bullish even if written neutrally.
3. Weight recent articles more heavily than older ones.
4. Return ONLY valid JSON matching the exact schema provided.
5. No markdown formatting inside JSON string values.
"""

ARTICLE_BATCH_SENTIMENT_PROMPT = """
Analyze the sentiment and key information from this batch of financial
news articles about {ticker} ({company_name}).

Articles:
{articles_text}

For each article, classify:
- sentiment: "bullish" | "bearish" | "neutral"
- confidence: 0.0-1.0 (how confident you are in the sentiment)
- key_point: one sentence capturing the most important information
- market_impact: "positive" | "negative" | "minimal" | "uncertain"
- themes: list of 1-3 themes from this list:
  [earnings, revenue, guidance, product_launch, regulation, legal,
   competition, macro, management, acquisition, partnership, analyst_rating,
   insider_activity, supply_chain, ai_technology, geopolitical]

Return this exact JSON (one entry per article, in same order):
{{
  "articles": [
    {{
      "article_index": 0,
      "sentiment": "bullish|bearish|neutral",
      "confidence": 0.0,
      "key_point": "one sentence",
      "market_impact": "positive|negative|minimal|uncertain",
      "themes": ["theme1", "theme2"]
    }}
  ]
}}
"""

THEME_ANALYSIS_PROMPT = """
Analyze the dominant themes and narratives across all classified news articles
about {ticker} ({company_name}).

Classified Articles Summary:
{classified_summary}

Theme Distribution:
{theme_counts}

Most Recent Headlines (last 7 days):
{recent_headlines}

Known Company Risks (from risk analysis):
{top_risks}

Evaluate:
1. What are the 3-5 dominant market narratives right now?
2. Is overall media coverage tone improving or deteriorating?
3. Which known company risks are being actively covered in the news?
4. Are there any emerging concerns NOT in the risk analysis?
5. What events or catalysts are being anticipated by the market?

Return this exact JSON:
{{
  "dominant_themes": [
    {{
      "theme": "theme name",
      "narrative": "2-3 sentence description of what the market is saying",
      "sentiment": "bullish|bearish|neutral",
      "article_count": 0,
      "intensity": "high|medium|low"
    }}
  ],
  "coverage_trend": "improving|stable|deteriorating",
  "coverage_trend_reasoning": "1-2 sentences",
  "risks_in_news": [
    {{
      "risk_name": "name from risk analysis",
      "coverage_level": "heavy|moderate|light|none",
      "sentiment": "bullish|bearish|neutral"
    }}
  ],
  "emerging_concerns": ["concern1", "concern2"],
  "anticipated_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
  "citation": ["headline1", "headline2"]
}}
"""

SENTIMENT_SYNTHESIS_PROMPT = """
Synthesize the complete sentiment analysis into a final market sentiment verdict.

Ticker: {ticker}
Company: {company_name}
Analysis Period: Last {lookback_days} days

Article Statistics:
- Total articles analyzed: {total_articles}
- Bullish: {bullish_count} ({bullish_pct}%)
- Bearish: {bearish_count} ({bearish_pct}%)
- Neutral: {neutral_count} ({neutral_pct}%)
- Weighted Sentiment Score: {weighted_score:.2f} (range: -1.0 to +1.0)

Dominant Themes:
{dominant_themes_text}

Emerging Concerns:
{emerging_concerns_text}

Anticipated Catalysts:
{catalysts_text}

Fundamental Verdict (Step 3): {fundamental_verdict}
Risk Verdict (Step 4): {risk_verdict}

Produce the final sentiment verdict the Lead Analyst will use.

Return this exact JSON:
{{
  "overall_sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
  "sentiment_score": 0.0,
  "sentiment_score_reasoning": "2-3 sentences explaining the score",
  "momentum": "accelerating_positive|stable_positive|neutral|stable_negative|accelerating_negative",
  "momentum_reasoning": "1-2 sentences",
  "news_vs_fundamentals_alignment": "aligned|diverging|conflicting",
  "alignment_reasoning": "does news sentiment match fundamental quality?",
  "key_bullish_signals": ["signal1", "signal2", "signal3"],
  "key_bearish_signals": ["signal1", "signal2"],
  "most_important_recent_event": "single most market-moving recent news item",
  "sentiment_verdict": "POSITIVE|NEUTRAL|NEGATIVE",
  "verdict_confidence": 0,
  "watch_for": ["upcoming event or data point to monitor 1",
                "upcoming event or data point to monitor 2"]
}}
"""
