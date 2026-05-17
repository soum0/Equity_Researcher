import glob
import json
import os

import streamlit as st


def render():
    # Hero section
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <h1 style="font-size:3em;">📈 AI Investment Research</h1>
        <p style="font-size:1.2em; color:#94a3b8;">
            Institutional-grade equity analysis powered by GraphRAG + Multi-Agent AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ticker input — centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ticker = st.text_input(
            "Enter Stock Ticker",
            placeholder="AAPL, MSFT, NVDA, TSLA...",
            help="Enter any NYSE or NASDAQ ticker symbol",
        ).upper().strip()

        # Persist ticker across reruns so button clicks don't lose it
        if ticker:
            st.session_state["ticker_input"] = ticker
        else:
            ticker = st.session_state.get("ticker_input", "")

        col_a, col_b = st.columns(2)
        with col_a:
            run_full = st.button("🚀 Full Analysis", use_container_width=True, type="primary")
        with col_b:
            view_existing = st.button("📋 View Results", use_container_width=True)

        if run_full and ticker:
            st.session_state["selected_ticker"] = ticker
            st.session_state["current_page"] = "🚀 Run Analysis"
            st.rerun()

        if view_existing and ticker:
            if os.path.exists(f"data/{ticker}/analysis/investment_memo.json"):
                st.session_state["selected_ticker"] = ticker
                st.session_state["current_page"] = "📋 Investment Memo"
                st.rerun()
            else:
                st.error(f"No analysis found for {ticker}. Run Full Analysis first.")

    st.markdown("---")

    # Pipeline overview
    st.markdown("### How It Works")
    cols = st.columns(7)
    steps = [
        ("📥", "Ingest",       "Price, News, SEC filings, Macro"),
        ("🕸️", "GraphRAG",    "Knowledge graph from 10-Ks"),
        ("🔍", "Fundamental", "Business quality & financials"),
        ("⚠️", "Risk",        "Bear case & risk chains"),
        ("📰", "Sentiment",   "96 news articles analyzed"),
        ("🎯", "Synthesis",   "Weighted BUY/HOLD/SELL"),
        ("📄", "Report",      "PDF investment memo"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:8px; background:#1e293b;
                        border-radius:8px; height:100px;">
                <div style="font-size:1.8em;">{icon}</div>
                <div style="font-weight:bold; font-size:0.85em;">{title}</div>
                <div style="font-size:0.7em; color:#94a3b8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Recent analyses
    memos = glob.glob("data/*/analysis/investment_memo.json")
    if memos:
        st.markdown("### Recent Analyses")
        cols = st.columns(min(len(memos), 4))
        for col, memo_path in zip(cols, memos[:4]):
            tkr = memo_path.split("/")[1]
            try:
                with open(memo_path) as f:
                    memo = json.load(f)
                rec    = memo.get("recommendation", "HOLD")
                conf   = memo.get("confidence", 0)
                target = memo.get("price_target_12m", 0)
                score  = memo.get("composite_score", 0)
                color  = "#22c55e" if rec == "BUY" else "#ef4444" if rec == "SELL" else "#eab308"
                with col:
                    st.markdown(f"""
                    <div style="background:#1e293b; border-radius:8px; padding:16px;
                                border-left: 4px solid {color};">
                        <div style="font-size:1.4em; font-weight:bold;">{tkr}</div>
                        <div style="color:{color}; font-weight:bold;">{rec}</div>
                        <div style="font-size:0.85em; color:#94a3b8;">
                            Target: ${target:.2f}<br>
                            Score: {score:.3f} | Conf: {conf}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"View {tkr}", key=f"view_{tkr}"):
                        st.session_state["selected_ticker"] = tkr
                        st.session_state["current_page"] = "📋 Investment Memo"
                        st.rerun()
            except Exception:
                pass
