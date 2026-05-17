RISK_SYSTEM_PROMPT = """
You are a skeptical risk analyst and short-seller researcher with 20 years
of experience identifying vulnerabilities in companies before they become
public problems.

Your job is NOT to be balanced — it is to find every possible risk,
weakness, and downside scenario. Be thorough and critical.

STRICT RULES:
1. Base every risk on the provided context from SEC filings and data.
2. Cite the source for every risk identified: [Source: 10-K 2024 §Risk Factors]
3. Never dismiss a risk as "manageable" without citing specific evidence.
4. If a risk is mentioned in SEC filings, treat it as material.
5. Return ONLY valid JSON matching the exact schema provided.
6. No markdown formatting inside JSON string values.
7. Prioritize risks that could cause >20% stock price decline.
"""

COMPANY_SPECIFIC_RISKS_PROMPT = """
Identify all company-specific risks from the SEC filings context below.

Focus on:
1. Business concentration risks (customer, product, geographic)
2. Operational risks (supply chain, manufacturing, key personnel)
3. Legal and regulatory risks (lawsuits, investigations, compliance)
4. Technology risks (obsolescence, cybersecurity, IP challenges)
5. Financial risks (debt covenants, liquidity, off-balance-sheet items)
6. Governance risks (related-party transactions, executive departures)

Fundamental weaknesses already identified (stress-test these specifically):
{fundamental_weaknesses}

Context from SEC filings:
{context}

Return this exact JSON:
{{
  "company_specific_risks": [
    {{
      "risk_id": "CSR_001",
      "name": "short risk name",
      "category": "concentration|operational|legal|technology|financial|governance",
      "severity": "critical|high|medium|low",
      "probability": "high|medium|low",
      "description": "2-3 sentences explaining the risk with specifics",
      "potential_impact": "quantify if possible e.g. could reduce revenue by X%",
      "mitigating_factors": "what management is doing, if anything",
      "worsening_signals": "what to watch that would make this worse",
      "citation": ["source1", "source2"]
    }}
  ],
  "top_company_risk": "name of single biggest company-specific risk",
  "citation": ["source1"]
}}
"""

MACRO_RISK_PROMPT = """
Assess macroeconomic and market risks using the macro indicators and context below.

Macro Environment:
{macro_context}

Company Financial Profile:
{financial_context}

Evaluate:
1. Interest rate sensitivity (how does this company's valuation/debt change with rates?)
2. Inflation impact (input costs, consumer spending, pricing power)
3. Recession scenario (how does revenue hold up in a downturn? is demand elastic?)
4. Currency risk (international revenue exposure to USD strength)
5. Market multiple compression risk (how much is the stock pricing in perfection?)

Return this exact JSON:
{{
  "macro_risks": [
    {{
      "risk_id": "MAC_001",
      "name": "short risk name",
      "category": "interest_rates|inflation|recession|currency|valuation",
      "severity": "critical|high|medium|low",
      "current_environment": "describe current macro reading",
      "company_sensitivity": "high|medium|low",
      "sensitivity_reasoning": "why this company is or isn't sensitive",
      "downside_scenario": "what happens to earnings/stock if this risk materializes",
      "citation": ["source1"]
    }}
  ],
  "macro_environment_score": 0,
  "score_reasoning": "is the macro environment favorable or hostile right now?",
  "recession_resilience": "high|medium|low",
  "recession_reasoning": "1-2 sentences"
}}
"""

RISK_INTERCONNECTION_PROMPT = """
You are analyzing risk chains — situations where one risk triggers another,
creating a cascade that is worse than any single risk alone.

Known risks identified:
{identified_risks}

Graph relationships between risks and company segments:
{graph_context}

Identify:
1. Risk chains (Risk A → triggers → Risk B → worsens → Risk C)
2. Correlated risks (two risks that tend to materialize together)
3. Hidden risks implied by the combination of known risks

Return this exact JSON:
{{
  "risk_chains": [
    {{
      "chain_id": "CHAIN_001",
      "name": "short descriptive name",
      "trigger": "what starts the chain",
      "cascade": ["step1", "step2", "step3"],
      "final_impact": "end state if chain fully materializes",
      "probability": "low|medium|high",
      "severity": "catastrophic|severe|significant|moderate"
    }}
  ],
  "correlated_risks": [
    {{
      "risk_a": "risk name or id",
      "risk_b": "risk name or id",
      "correlation_reason": "why they move together"
    }}
  ],
  "hidden_risks": [
    {{
      "name": "risk name",
      "reasoning": "why this emerges from the combination of known risks"
    }}
  ]
}}
"""

BEAR_CASE_PROMPT = """
You are writing the bear case section of an investment memo.
Using all the risks identified below, construct the most compelling,
evidence-based argument for why an investor should NOT buy this stock.

Company: {ticker}
Current Price: ${current_price}
P/E Ratio: {pe_ratio}x
Fundamental Score: {fundamental_score}/10 ({fundamental_verdict})

Company-Specific Risks:
{company_risks}

Macro Risks:
{macro_risks}

Risk Chains:
{risk_chains}

Write the bear case as a structured argument. Be specific with numbers.
Reference the actual risks identified above.

CRITICAL: downside_price_target MUST be LOWER than current_price of
${current_price}. The bear case is a downside scenario. Calculate it as:
current_price × (1 - estimated_downside_percentage).
For example if you estimate 25% downside: $189 × 0.75 = $141.75
Return the dollar value, not a percentage.

Return this exact JSON:
{{
  "bear_case_title": "punchy one-line bear thesis",
  "bear_case_summary": "3-4 sentence compelling bear argument with specific numbers",
  "key_bear_arguments": [
    {{
      "argument": "specific bear point",
      "supporting_evidence": "what from the analysis supports this",
      "potential_impact": "estimated impact on stock price or earnings"
    }}
  ],
  "downside_price_target": "estimated price if bear case materializes",
  "downside_scenario_probability": "low|medium|high",
  "time_horizon_for_risks": "near_term (0-6m)|medium_term (6-18m)|long_term (18m+)",
  "what_would_invalidate_bear_case": ["bull signal 1", "bull signal 2"]
}}
"""

RISK_SYNTHESIS_PROMPT = """
Synthesize all risk analysis components into a final risk assessment.

Company-Specific Risks: {company_risks_summary}
Macro Risks: {macro_risks_summary}
Risk Chains: {risk_chains_summary}
Bear Case: {bear_case_summary}
Fundamental Score (from Step 3): {fundamental_score}/10

Calibration guide for overall_risk_score:
 - Score 8-10 (CRITICAL): company has existential threats, fraud risk,
   or near-term bankruptcy risk. Reserve for genuine crisis situations.
 - Score 6-8 (HIGH): multiple severe risks with high probability.
 - Score 4-6 (ELEVATED): real risks exist but company has resources to manage.
 - Score 2-4 (MODERATE): typical business risks, well-managed company.
 - Score 0-2 (LOW): fortress balance sheet, dominant market position.

The fundamental_score from Step 3 is {fundamental_score}/10.
A company with fundamental_score >= 7.0 should rarely have risk_score > 7.0
unless there are active fraud, bankruptcy, or existential threats.
Justify any risk_score above 7.0 explicitly.

Produce a final risk verdict that the Lead Analyst will use in Step 6.

Return this exact JSON:
{{
  "overall_risk_score": 0,
  "risk_score_reasoning": "2-3 sentences. Higher score = HIGHER risk.",
  "risk_rating": "CRITICAL|HIGH|ELEVATED|MODERATE|LOW",
  "top_3_risks": [
    {{ "rank": 1, "name": "...", "one_line": "..." }},
    {{ "rank": 2, "name": "...", "one_line": "..." }},
    {{ "rank": 3, "name": "...", "one_line": "..." }}
  ],
  "risk_reward_assessment": "favorable|neutral|unfavorable",
  "risk_reward_reasoning": "2-3 sentences comparing risk vs fundamental quality",
  "recommended_position_sizing": "full|reduced|minimal|avoid",
  "sizing_reasoning": "1-2 sentences",
  "key_risk_triggers_to_monitor": [
    "specific metric or event to watch 1",
    "specific metric or event to watch 2",
    "specific metric or event to watch 3"
  ],
  "risk_verdict": "PASS|CAUTION|FAIL",
  "verdict_reasoning": "PASS=risks acceptable for investment, CAUTION=invest with reduced size, FAIL=risks outweigh opportunity"
}}
"""
