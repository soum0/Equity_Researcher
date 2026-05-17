import glob
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st

# Page config — MUST be the first Streamlit call
st.set_page_config(
    page_title="AI Investment Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown("""
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* Recommendation boxes */
    .rec-box-buy  { background: #166534; border: 2px solid #22c55e;
                    border-radius: 12px; padding: 20px; text-align: center; }
    .rec-box-hold { background: #713f12; border: 2px solid #eab308;
                    border-radius: 12px; padding: 20px; text-align: center; }
    .rec-box-sell { background: #7f1d1d; border: 2px solid #ef4444;
                    border-radius: 12px; padding: 20px; text-align: center; }

    /* Score cards */
    .score-card { background: #1e293b; border-radius: 8px;
                  padding: 16px; text-align: center; }

    /* Metric container override */
    [data-testid="metric-container"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    </style>
    """, unsafe_allow_html=True)


inject_css()

from streamlit_pages import data_explorer, home, memo_viewer, pipeline_runner

PAGES = {
    "🏠 Home":            home,
    "🚀 Run Analysis":    pipeline_runner,
    "📋 Investment Memo": memo_viewer,
    "🔍 Data Explorer":   data_explorer,
}

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📈 AI Investment Research")
st.sidebar.markdown("---")

# Quick-access buttons for already-analyzed tickers
analyzed = sorted(
    d.split(os.sep)[1]
    for d in glob.glob(os.path.join("data", "*", "analysis", "investment_memo.json"))
)
if analyzed:
    st.sidebar.markdown("**📊 Analyzed Tickers**")
    for t in analyzed:
        if st.sidebar.button(t, key=f"sidebar_{t}"):
            st.session_state["selected_ticker"] = t
            st.session_state["current_page"] = "📋 Investment Memo"
            st.rerun()

st.sidebar.markdown("---")

# Get current page index from session state
current = st.session_state.get("current_page", "🏠 Home")
if current not in PAGES:
    current = "🏠 Home"
default_idx = list(PAGES.keys()).index(current)

page_name = st.sidebar.radio("Navigate", list(PAGES.keys()),
                              index=default_idx, key="nav")
# Only update if USER changed it (not programmatic navigation)
st.session_state["current_page"] = page_name

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
AI-generated research. Not financial advice.<br>
Built with LangGraph + GraphRAG + Groq.
</small>
""", unsafe_allow_html=True)

# ── Render selected page ───────────────────────────────────────────────────
PAGES[page_name].render()
