import plotly.graph_objects as go
import streamlit as st


def render_score_charts(memo: dict, fundamental: dict, risk: dict, sentiment: dict):
    """Render the 3-component score breakdown as Plotly charts."""
    breakdown = memo.get("score_breakdown", {})
    f_score   = breakdown.get("fundamental_score", 0)
    r_score   = breakdown.get("risk_score", 0)
    s_score   = float(breakdown.get("sentiment_score", 0))
    composite = float(memo.get("composite_score", 0))
    rec       = memo.get("recommendation", "HOLD")

    col1, col2, col3, col4 = st.columns(4)

    _layout = dict(
        height=220,
        margin=dict(t=60, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )

    # ── Fundamental gauge ─────────────────────────────────────────
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=f_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fundamental<br><sub>Quality (0-10)</sub>",
                   "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar":  {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 4],  "color": "#7f1d1d"},
                    {"range": [4, 7],  "color": "#713f12"},
                    {"range": [7, 10], "color": "#166534"},
                ],
                "threshold": {"line": {"color": "white", "width": 2},
                              "thickness": 0.75, "value": f_score},
            },
        ))
        fig.update_layout(**_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Component: {breakdown.get('fundamental_component', 0):.3f} × 0.40")

    # ── Risk gauge (lower = safer) ────────────────────────────────
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score<br><sub>Lower = Safer (0-10)</sub>",
                   "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar":  {"color": "#ef4444"},
                "steps": [
                    {"range": [0, 4],  "color": "#166534"},
                    {"range": [4, 7],  "color": "#713f12"},
                    {"range": [7, 10], "color": "#7f1d1d"},
                ],
            },
        ))
        fig.update_layout(**_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Component: {breakdown.get('risk_component', 0):.3f} × 0.35 (inverted)")

    # ── Sentiment gauge ───────────────────────────────────────────
    with col3:
        color = "#22c55e" if s_score > 0.1 else "#ef4444" if s_score < -0.1 else "#eab308"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s_score,
            number={"suffix": "", "valueformat": ".2f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Sentiment<br><sub>Score (-1 to +1)</sub>",
                   "font": {"size": 14}},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [-1, -0.3],  "color": "#7f1d1d"},
                    {"range": [-0.3, 0.3], "color": "#713f12"},
                    {"range": [0.3, 1],    "color": "#166534"},
                ],
            },
        ))
        fig.update_layout(**_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Component: {breakdown.get('sentiment_component', 0):.3f} × 0.25")

    # ── Composite waterfall ───────────────────────────────────────
    with col4:
        f_contrib   = float(breakdown.get("fundamental_component", 0)) * 0.40
        r_contrib   = float(breakdown.get("risk_component", 0)) * 0.35
        s_contrib   = float(breakdown.get("sentiment_component", 0)) * 0.25
        rec_color   = "#22c55e" if rec == "BUY" else "#ef4444" if rec == "SELL" else "#eab308"

        fig = go.Figure(go.Waterfall(
            name="Score",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Fund.", "Risk", "Sent.", "Total"],
            y=[f_contrib, r_contrib, s_contrib, 0],
            connector={"line": {"color": "#475569"}},
            increasing={"marker": {"color": "#3b82f6"}},
            totals={"marker": {"color": rec_color}},
            text=[f"{f_contrib:.3f}", f"{r_contrib:.3f}",
                  f"{s_contrib:.3f}", f"{composite:.3f}"],
            textposition="outside",
        ))
        fig.add_hline(y=0.65, line_dash="dash", line_color="#22c55e",
                      annotation_text="BUY threshold")
        fig.add_hline(y=0.45, line_dash="dash", line_color="#eab308",
                      annotation_text="HOLD threshold")
        fig.update_layout(
            title="Composite Build-up",
            height=220,
            margin=dict(t=40, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f1f5f9"},
            yaxis={"range": [0, 1], "gridcolor": "#334155"},
            xaxis={"gridcolor": "#334155"},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"→ {rec} @ {composite:.3f}")
