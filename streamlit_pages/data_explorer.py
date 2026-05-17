import json
import os

import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=300)
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def render():
    st.markdown("## 🔍 Data Explorer")

    ticker = st.session_state.get("selected_ticker", "")
    if not ticker:
        st.info("Select a ticker first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📰 News", "📄 SEC Filings", "🕸️ Graph", "📈 Price"]
    )

    # ── News ─────────────────────────────────────────────────────────────────
    with tab1:
        news_path = f"data/{ticker}/news_articles.json"
        if not os.path.exists(news_path):
            st.info("News data not found. Run ingestion first.")
        else:
            news     = _load_json(news_path)
            articles = news.get("articles", [])
            st.markdown(f"**{len(articles)} articles** over last 30 days")

            sentiment_path = f"data/{ticker}/analysis/sentiment_analysis.json"
            classified_map: dict = {}
            if os.path.exists(sentiment_path):
                sent = _load_json(sentiment_path)
                bd   = sent.get("article_breakdown", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("🐂 Bullish", f"{bd.get('bullish', 0)} ({bd.get('bullish_pct', 0):.0f}%)")
                col2.metric("😐 Neutral", str(bd.get("neutral", 0)))
                col3.metric("🐻 Bearish", f"{bd.get('bearish', 0)} ({bd.get('bearish_pct', 0):.0f}%)")
                classified_map = {
                    a.get("url", ""): a
                    for a in sent.get("classified_articles", [])
                }

            for art in articles[:20]:
                label = classified_map.get(art.get("url", ""), {}).get("sentiment_label", "")
                color = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "⚪"
                headline = (art.get("headline") or "")[:80]
                with st.expander(f"{color} {headline}..."):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(art.get("summary", ""))
                    with col2:
                        st.caption(f"**Source:** {art.get('source', '')}")
                        st.caption(f"**Date:** {(art.get('published_at') or '')[:10]}")
                        if art.get("url"):
                            st.markdown(f"[Read more →]({art['url']})")

    # ── SEC Filings ───────────────────────────────────────────────────────────
    with tab2:
        meta_path = f"data/{ticker}/sec_filings/metadata.json"
        if not os.path.exists(meta_path):
            st.info("SEC filing metadata not found. Run ingestion first.")
        else:
            meta    = _load_json(meta_path)
            filings = meta.get("filings", [])
            st.markdown(f"**{len(filings)} SEC filings** downloaded")
            for filing in filings:
                with st.expander(
                    f"📄 {filing.get('form_type', '')} — {filing.get('filing_date', '')}"
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Size",     f"{filing.get('char_count', 0):,} chars")
                    col2.metric("Sections", len(filing.get("sections_detected", [])))
                    col3.metric("Form",     filing.get("form_type", ""))
                    sections = filing.get("sections_detected", [])
                    if sections:
                        st.markdown("**Sections detected:** " + ", ".join(sections))

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    with tab3:
        report_path = f"data/{ticker}/graphrag/graphrag_report.json"
        if not os.path.exists(report_path):
            st.info("GraphRAG index not found. Run --build-graph first.")
        else:
            report = _load_json(report_path)
            graph  = report.get("graph", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nodes",     graph.get("total_nodes", 0))
            col2.metric("Edges",     graph.get("total_edges", 0))
            col3.metric("Entities",  report.get("extraction", {}).get("total_entities", 0))
            col4.metric("Vectors",   report.get("vector_store", {}).get("total_vectors", 0))

            node_types = graph.get("nodes_by_type", {})
            if node_types:
                fig = go.Figure(go.Pie(
                    labels=list(node_types.keys()),
                    values=list(node_types.values()),
                    hole=0.4,
                    textinfo="label+percent",
                ))
                fig.update_layout(
                    title="Knowledge Graph — Node Types",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f1f5f9"},
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Price History ─────────────────────────────────────────────────────────
    with tab4:
        price_path = f"data/{ticker}/price_data.json"
        if not os.path.exists(price_path):
            st.info("Price data not found. Run ingestion first.")
        else:
            price_data = _load_json(price_path)
            history    = price_data.get("price_history", {}).get("history", [])
            if history:
                import pandas as pd

                df = pd.DataFrame(history)
                df["date"] = pd.to_datetime(df["date"])
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=ticker,
                ))
                fig.update_layout(
                    title=f"{ticker} — 2Y Price History",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,1)",
                    font={"color": "#f1f5f9"},
                    xaxis={"gridcolor": "#334155"},
                    yaxis={"gridcolor": "#334155"},
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                summary = price_data.get("price_history", {}).get("summary", {})
                if summary:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Latest Close",   f"${summary.get('latest_close', 0):.2f}")
                    col2.metric("52W High",        f"${summary.get('52w_high', 0):.2f}")
                    col3.metric("52W Low",         f"${summary.get('52w_low', 0):.2f}")
                    col4.metric("1Y Return",       f"{summary.get('price_change_1y_pct', 0):+.1f}%")
            else:
                st.info("No price history available.")
