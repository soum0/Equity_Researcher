import json
import os

import streamlit as st

from streamlit_pages.components.file_downloader import render_pdf_download
from streamlit_pages.components.recommendation_box import render_recommendation_box
from streamlit_pages.components.score_charts import render_score_charts


@st.cache_data(ttl=300)
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def render():
    st.markdown("## 📋 Investment Memo")

    ticker = st.session_state.get("selected_ticker", "")
    if not ticker:
        st.info("Select a ticker from the sidebar or run an analysis first.")
        return

    memo_path = f"data/{ticker}/analysis/investment_memo.json"
    if not os.path.exists(memo_path):
        st.error(f"No memo found for {ticker}. Run the pipeline first.")
        return

    memo        = _load_json(memo_path)
    fundamental = _load_json(f"data/{ticker}/analysis/fundamental_analysis.json")
    risk        = _load_json(f"data/{ticker}/analysis/risk_analysis.json")
    sentiment   = _load_json(f"data/{ticker}/analysis/sentiment_analysis.json")

    # ── Header ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        company   = memo.get("company_name", ticker)
        generated = (memo.get("generated_at") or "")[:10]
        st.markdown(f"### {company} ({ticker})")
        st.caption(f"Generated: {generated} | Model: AI Research Pipeline v1.0")
    with col2:
        render_pdf_download(ticker)

    st.markdown("---")

    # ── Recommendation box ────────────────────────────────────────────────
    render_recommendation_box(memo)
    st.markdown("---")

    # ── Score breakdown charts ────────────────────────────────────────────
    st.markdown("### 📊 Score Breakdown")
    render_score_charts(memo, fundamental, risk, sentiment)
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Memo", "🐂 Bull Case", "🐻 Bear Case", "⚠️ Risks", "📋 Checklist",
    ])

    memo_inner = memo.get("memo") or {}

    with tab1:
        st.markdown("#### Executive Summary")
        st.markdown(memo_inner.get("executive_summary") or memo.get("executive_summary", ""))

        st.markdown("#### Investment Thesis")
        st.markdown(memo_inner.get("investment_thesis") or memo.get("investment_thesis", ""))

        st.markdown("#### Key Financial Metrics")
        metrics = memo_inner.get("key_financial_metrics") or {}
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price",  f"${metrics.get('current_price', 0):.2f}")
        col2.metric("Price Target",   f"${metrics.get('price_target_12m', 0):.2f}",
                    f"{metrics.get('upside_downside_pct', 0):+.1f}%")
        col3.metric("P/E Ratio",      f"{metrics.get('pe_ratio', 0):.1f}x")
        col4.metric("Net Margin",     f"{metrics.get('net_margin_pct', 0):.1f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("FCF",            f"${metrics.get('fcf_billions', 0):.1f}B")
        col2.metric("Rev Growth YoY", f"{metrics.get('revenue_growth_yoy_pct', 0):.1f}%")
        col3.metric("Confidence",     f"{memo.get('confidence', 0)}%")
        col4.metric("Composite",      f"{memo.get('composite_score', 0):.3f}")

        st.markdown("#### Key Drivers")
        for driver in memo.get("key_drivers", []):
            st.markdown(f"✅ {driver}")

        notes = memo_inner.get("analyst_notes") or ""
        if notes:
            st.markdown("#### Analyst Notes")
            st.info(notes)

    with tab2:
        bull = memo.get("bull_case") or {}
        st.markdown(f"### 🐂 {bull.get('title', '')}")
        st.success(bull.get("thesis", ""))

        st.markdown("**Key Drivers:**")
        for d in bull.get("key_drivers", []):
            if isinstance(d, dict):
                st.markdown(f"**{d.get('driver', '')}:** {d.get('detail', '')}")
            else:
                st.markdown(f"• {d}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Upside Scenario:**")
            st.markdown(bull.get("upside_scenario", ""))
        with col2:
            st.metric("Bull Price Target", str(bull.get("price_target_bull", "")))

    with tab3:
        bear = memo.get("bear_case") or {}
        st.markdown(f"### 🐻 {bear.get('title', '')}")
        st.error(bear.get("thesis", ""))

        st.markdown("**Key Risks:**")
        for r in bear.get("key_risks", []):
            if isinstance(r, dict):
                st.markdown(f"**{r.get('risk', '')}:** {r.get('detail', '')}")
            else:
                st.markdown(f"• {r}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Downside Scenario:**")
            st.markdown(bear.get("downside_scenario", ""))
        with col2:
            st.metric("Bear Price Target", str(bear.get("price_target_bear", "")))

    with tab4:
        st.markdown("### ⚠️ Key Risks")
        for i, risk_item in enumerate(memo.get("key_risks", []), 1):
            st.markdown(f"**{i}.** {risk_item}")

        if risk:
            st.markdown("### 🔗 Risk Chains")
            chains = (risk.get("risk_interconnections") or {}).get("risk_chains", [])
            for chain in chains:
                with st.expander(
                    f"🔗 {chain.get('name', '')} ({chain.get('severity', '')})"
                ):
                    st.markdown(f"**Trigger:** {chain.get('trigger', '')}")
                    for step in chain.get("cascade", []):
                        st.markdown(f"  → {step}")
                    st.markdown(f"**Final Impact:** {chain.get('final_impact', '')}")

    with tab5:
        st.markdown("### 📋 Monitoring Checklist")
        for item in memo.get("monitoring_checklist", []):
            st.checkbox(item, key=f"check_{item[:30]}")

        catalysts = memo.get("catalysts", [])
        if catalysts:
            st.markdown("### 🔭 Catalysts to Watch")
            for item in catalysts:
                st.markdown(f"→ {item}")

    st.markdown("---")
    st.caption("⚠️ AI-generated research for educational purposes only. Not financial advice.")
