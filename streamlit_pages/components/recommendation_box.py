import streamlit as st


def render_recommendation_box(memo: dict):
    rec       = memo.get("recommendation", "HOLD")
    conf      = memo.get("confidence", 0)
    target    = memo.get("price_target_12m", 0)
    upside    = memo.get("expected_return_pct", 0)
    composite = memo.get("composite_score", 0)

    color_map = {"BUY": "#22c55e", "HOLD": "#eab308", "SELL": "#ef4444"}
    bg_map    = {"BUY": "#052e16", "HOLD": "#1c1008", "SELL": "#1c0a0a"}
    icon_map  = {"BUY": "⭐", "HOLD": "◆", "SELL": "▼"}

    color = color_map.get(rec, "#eab308")
    bg    = bg_map.get(rec, "#1c1008")
    icon  = icon_map.get(rec, "◆")

    st.markdown(f"""
    <div style="background:{bg}; border:2px solid {color}; border-radius:12px;
                padding:24px; text-align:center; margin:16px 0;">
        <div style="font-size:3em; font-weight:bold; color:{color};">
            {icon} {rec} {icon}
        </div>
        <div style="display:flex; justify-content:center; gap:40px; margin-top:12px;">
            <div>
                <div style="color:#94a3b8; font-size:0.85em;">Confidence</div>
                <div style="font-size:1.4em; font-weight:bold;">{conf}%</div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.85em;">12M Price Target</div>
                <div style="font-size:1.4em; font-weight:bold;">${target:.2f}</div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.85em;">Expected Return</div>
                <div style="font-size:1.4em; font-weight:bold; color:{color};">
                    {upside:+.1f}%
                </div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.85em;">Composite Score</div>
                <div style="font-size:1.4em; font-weight:bold;">{composite:.3f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
