FUNDAMENTAL_SYSTEM_PROMPT = """
You are a senior equity research analyst with CFA designation and 15 years
of experience analyzing technology and growth companies.

Your task is to perform fundamental analysis on a company using ONLY the
provided context from SEC filings, financial data, and knowledge graph.

STRICT RULES:
1. Base EVERY claim on the provided context. Never use outside knowledge.
2. Cite the source for every factual claim: [Source: 10-K 2024 §Risk Factors]
3. Use specific numbers from the context wherever possible.
4. If context is insufficient to answer a sub-question, write "INSUFFICIENT DATA"
   rather than guessing.
5. Be critical and balanced — identify both strengths and weaknesses.
6. Return ONLY valid JSON matching the exact schema provided.
7. No markdown formatting inside JSON string values.
"""

BUSINESS_QUALITY_PROMPT = """
Analyze the company's core business quality using the context below.

Evaluate:
1. What does the company actually do? What are its main business segments?
2. What is its competitive moat? (pricing power, switching costs, network effects,
   cost advantages, intangible assets)
3. How sustainable is the revenue? (recurring vs one-time, diversified vs concentrated)
4. Who are the key customers and how dependent is the company on them?
5. What are the main growth drivers cited in the filings?

Context:
{context}

Return this exact JSON:
{{
  "business_description": "2-3 sentence plain English description",
  "segments": [
    {{
      "name": "segment name",
      "revenue_contribution": "% or USD if available",
      "growth_trend": "growing/stable/declining",
      "description": "one sentence"
    }}
  ],
  "moat_type": ["pricing_power", "switching_costs", "network_effects",
                "cost_advantage", "intangible_assets", "none"],
  "moat_strength": "wide/narrow/none",
  "moat_reasoning": "2-3 sentences with citations",
  "revenue_quality": "high/medium/low",
  "revenue_quality_reasoning": "explanation with citations",
  "key_growth_drivers": ["driver1", "driver2", "driver3"],
  "customer_concentration_risk": "high/medium/low",
  "citation": ["source1", "source2"]
}}
"""

FINANCIAL_HEALTH_PROMPT = """
Analyze the company's financial health and quality of earnings using the context below.

Evaluate:
1. Profitability trend (margins: gross, operating, net — improving or declining?)
2. Balance sheet strength (debt levels, cash position, liquidity)
3. Cash flow quality (is FCF growing? Is earnings backed by cash?)
4. Capital allocation (buybacks, dividends, R&D, acquisitions)
5. Key valuation ratios vs typical industry ranges

Context:
{context}

Return this exact JSON:
{{
  "profitability": {{
    "gross_margin": null,
    "operating_margin": null,
    "net_margin": null,
    "margin_trend": "expanding/stable/contracting",
    "roe": null,
    "roic": null,
    "assessment": "one sentence"
  }},
  "balance_sheet": {{
    "cash_and_equivalents": null,
    "total_debt": null,
    "debt_to_equity": null,
    "current_ratio": null,
    "strength": "strong/adequate/weak",
    "assessment": "one sentence"
  }},
  "cash_flow": {{
    "free_cash_flow": null,
    "fcf_margin": null,
    "fcf_trend": "growing/stable/declining",
    "earnings_quality": "high/medium/low",
    "assessment": "one sentence"
  }},
  "capital_allocation": {{
    "buybacks_active": true,
    "dividend_yield": null,
    "rd_as_pct_revenue": null,
    "assessment": "one sentence"
  }},
  "valuation": {{
    "pe_ratio": null,
    "ev_ebitda": null,
    "ps_ratio": null,
    "assessment": "cheap/fair/expensive relative to quality"
  }},
  "citation": ["source1", "source2"]
}}
"""

MANAGEMENT_QUALITY_PROMPT = """
Evaluate management quality and corporate governance using the context below.

Look for:
1. Track record: have they delivered on past guidance?
2. Capital allocation discipline (do they overpay for acquisitions? excessive dilution?)
3. Insider activity (buying or selling their own stock?)
4. Executive compensation structure (aligned with shareholders?)
5. Any red flags: restatements, SEC investigations, excessive perks

Context:
{context}

Return this exact JSON:
{{
  "ceo": "name if available",
  "tenure_assessment": "experienced/new/unknown",
  "guidance_track_record": "consistent/mixed/poor/unknown",
  "insider_sentiment": "bullish/neutral/bearish/unknown",
  "insider_reasoning": "explanation",
  "compensation_aligned": true,
  "red_flags": ["flag1", "flag2"],
  "overall_management_quality": "high/medium/low",
  "assessment": "2-3 sentences",
  "citation": ["source1", "source2"]
}}
"""

COMPETITIVE_POSITION_PROMPT = """
Assess the company's competitive position in its industry using the context below.

Evaluate:
1. Who are the main competitors?
2. What is the company's market position (leader/challenger/niche)?
3. Are there threats from new entrants or substitutes?
4. How is the competitive landscape changing (AI disruption, regulation, new tech)?
5. What does the company say about competition in its filings?

Context:
{context}

Return this exact JSON:
{{
  "market_position": "leader/strong_challenger/challenger/niche",
  "main_competitors": ["competitor1", "competitor2", "competitor3"],
  "competitive_advantages": ["advantage1", "advantage2"],
  "competitive_threats": ["threat1", "threat2"],
  "disruption_risk": "high/medium/low",
  "disruption_reasoning": "explanation with citations",
  "industry_tailwinds": ["tailwind1", "tailwind2"],
  "industry_headwinds": ["headwind1", "headwind2"],
  "citation": ["source1", "source2"]
}}
"""

SYNTHESIS_PROMPT = """
You are a senior equity analyst writing the fundamental analysis section of
an investment memo.

Using the structured analysis components below, synthesize a comprehensive
fundamental assessment with an overall score and investment thesis.

Weight the score as follows:
- financial_health: 35%
- business_quality: 30%
- competitive_position: 25%
- management_quality: 10%

Business Quality Analysis:
{business_quality}

Financial Health Analysis:
{financial_health}

Management Quality Analysis:
{management_quality}

Competitive Position Analysis:
{competitive_position}

Return this exact JSON:
{{
  "overall_fundamental_score": 0,
  "score_reasoning": "2-3 sentences explaining the score",
  "investment_thesis_long": "3-4 sentence bull case based on fundamentals",
  "key_fundamental_strengths": ["strength1", "strength2", "strength3"],
  "key_fundamental_weaknesses": ["weakness1", "weakness2", "weakness3"],
  "earnings_quality_verdict": "high/medium/low",
  "business_predictability": "high/medium/low",
  "recommended_further_research": ["area1", "area2"],
  "fundamental_verdict": "BUY_LEANING/HOLD_LEANING/SELL_LEANING",
  "verdict_confidence": 0,
  "analyst_notes": "Any important caveats or data gaps"
}}
"""
