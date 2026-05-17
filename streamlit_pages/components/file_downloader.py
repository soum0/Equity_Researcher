import os

import streamlit as st


def render_pdf_download(ticker: str):
    pdf_path = f"data/{ticker}/analysis/investment_memo.pdf"
    md_path  = f"data/{ticker}/analysis/investment_memo.md"

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF",
                data=f.read(),
                file_name=f"{ticker}_investment_memo.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            st.download_button(
                label="⬇️ Download Markdown",
                data=f.read(),
                file_name=f"{ticker}_investment_memo.md",
                mime="text/markdown",
                use_container_width=True,
            )
