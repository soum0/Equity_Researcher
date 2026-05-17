from __future__ import annotations

import os
from datetime import datetime


class PDFGenerator:
    """
    Generates a professional PDF investment memo from the markdown + JSON.
    Uses reportlab (pure Python, no external tools needed).
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.output_path = f"data/{ticker}/analysis/investment_memo.pdf"

    def generate(self, memo_md: str, memo_json: dict) -> str:
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        # ── Color scheme ──────────────────────────────────────────────
        BUY_COLOR  = HexColor("#22c55e")
        HOLD_COLOR = HexColor("#eab308")
        SELL_COLOR = HexColor("#ef4444")
        HEADER_BG  = HexColor("#1e293b")
        ACCENT     = HexColor("#3b82f6")
        LIGHT_GRAY = HexColor("#f1f5f9")
        MED_GRAY   = HexColor("#94a3b8")

        rec = memo_json.get("recommendation", "HOLD")
        rec_color = BUY_COLOR if rec == "BUY" else (SELL_COLOR if rec == "SELL" else HOLD_COLOR)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=f"Equity Research — {self.ticker}",
            author="AI Financial Analysis Pipeline",
        )

        # ── Styles ────────────────────────────────────────────────────
        base = getSampleStyleSheet()
        h1 = ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=18, textColor=white, spaceAfter=4,
            fontName="Helvetica-Bold",
        )
        h2 = ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=12, textColor=ACCENT, spaceAfter=4, spaceBefore=12,
            fontName="Helvetica-Bold",
        )
        h3 = ParagraphStyle(
            "H3", parent=base["Heading3"],
            fontSize=10, textColor=HEADER_BG, spaceAfter=3, spaceBefore=8,
            fontName="Helvetica-Bold",
        )
        body = ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9, leading=13, spaceAfter=6,
        )
        small = ParagraphStyle(
            "Small", parent=base["Normal"],
            fontSize=7.5, textColor=MED_GRAY, leading=11,
        )
        bullet = ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9, leading=13, leftIndent=12, spaceAfter=3,
            bulletIndent=0,
        )

        story = []

        # ── Header banner ─────────────────────────────────────────────
        company_name = memo_json.get("company_name", self.ticker)
        sector = ""
        date_str = datetime.now().strftime("%B %d, %Y")

        header_data = [[
            Paragraph(f"EQUITY RESEARCH — {self.ticker}", h1),
            Paragraph(
                f'<font color="white">{company_name}<br/>Generated: {date_str}</font>',
                ParagraphStyle("sub", parent=h1, fontSize=9, fontName="Helvetica"),
            ),
        ]]
        header_table = Table(header_data, colWidths=["60%", "40%"])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",    (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.3 * cm))

        # ── Recommendation box ────────────────────────────────────────
        story.append(self._recommendation_box(memo_json, rec_color, body))
        story.append(Spacer(1, 0.3 * cm))

        # ── Scorecard ─────────────────────────────────────────────────
        story.append(Paragraph("SCORECARD", h2))
        story.append(self._scorecard_table(memo_json))
        story.append(Spacer(1, 0.3 * cm))

        # ── Executive Summary ─────────────────────────────────────────
        story.append(Paragraph("EXECUTIVE SUMMARY", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
        story.append(Spacer(1, 0.1 * cm))
        exec_summary = memo_json.get("memo", {}).get("executive_summary") or memo_json.get("executive_summary", "")
        story.append(Paragraph(exec_summary, body))
        story.append(Spacer(1, 0.2 * cm))

        # ── Investment Thesis ─────────────────────────────────────────
        story.append(Paragraph("INVESTMENT THESIS", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
        story.append(Spacer(1, 0.1 * cm))
        thesis = memo_json.get("investment_thesis") or memo_json.get("memo", {}).get("investment_thesis", "")
        story.append(Paragraph(thesis, body))
        story.append(Spacer(1, 0.2 * cm))

        # ── Key Financial Metrics ─────────────────────────────────────
        story.append(Paragraph("KEY FINANCIAL METRICS", h2))
        story.append(self._metrics_table(memo_json))
        story.append(Spacer(1, 0.3 * cm))

        # ── Bull / Bear Cases ─────────────────────────────────────────
        story.append(Paragraph("BULL CASE vs BEAR CASE", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
        story.append(Spacer(1, 0.1 * cm))
        story.append(self._bull_bear_table(memo_json, body, BUY_COLOR, SELL_COLOR, LIGHT_GRAY))
        story.append(Spacer(1, 0.3 * cm))

        # ── Key Risks / Catalysts side by side ───────────────────────
        risks    = memo_json.get("key_risks", [])
        catalysts = memo_json.get("catalysts", [])
        story.append(Paragraph("TOP RISKS  &  CATALYSTS", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
        story.append(Spacer(1, 0.1 * cm))
        max_rows = max(len(risks), len(catalysts), 1)
        rc_data = [["TOP RISKS", "CATALYSTS"]]
        for i in range(max_rows):
            r = f"• {risks[i]}" if i < len(risks) else ""
            c = f"• {catalysts[i]}" if i < len(catalysts) else ""
            rc_data.append([Paragraph(r, body), Paragraph(c, body)])
        rc_table = Table(rc_data, colWidths=["50%", "50%"])
        rc_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
            ("TEXTCOLOR",   (0, 0), (-1, 0), white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("PADDING",     (0, 0), (-1, -1), 6),
        ]))
        story.append(rc_table)
        story.append(Spacer(1, 0.3 * cm))

        # ── Monitoring Checklist ──────────────────────────────────────
        checklist = memo_json.get("monitoring_checklist", [])
        if checklist:
            story.append(Paragraph("MONITORING CHECKLIST", h2))
            story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
            story.append(Spacer(1, 0.1 * cm))
            for item in checklist:
                story.append(Paragraph(f"☐  {item}", bullet))
            story.append(Spacer(1, 0.2 * cm))

        # ── Analyst Notes ─────────────────────────────────────────────
        analyst_notes = memo_json.get("memo", {}).get("analyst_notes") or memo_json.get("analyst_notes", "")
        if analyst_notes:
            story.append(Paragraph("ANALYST NOTES", h2))
            story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
            story.append(Spacer(1, 0.1 * cm))
            story.append(Paragraph(analyst_notes, body))
            story.append(Spacer(1, 0.2 * cm))

        # ── Disclaimer footer ─────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5, color=MED_GRAY))
        story.append(Spacer(1, 0.1 * cm))
        story.append(Paragraph(
            "Generated by AI Financial Analysis Pipeline  |  "
            f"{date_str}  |  "
            "This is AI-generated research for educational purposes only. "
            "Not financial advice. Always conduct your own due diligence.",
            small,
        ))

        doc.build(story)
        return self.output_path

    def _recommendation_box(self, memo_json: dict, rec_color, body_style) -> Table:
        from reportlab.lib.colors import white
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        rec          = memo_json.get("recommendation", "HOLD")
        confidence   = memo_json.get("confidence", 0)
        price_target = memo_json.get("price_target_12m", 0)
        composite    = memo_json.get("composite_score", 0)
        ret_pct      = memo_json.get("expected_return_pct", 0)
        current      = memo_json.get("current_price", 0)

        rec_style = ParagraphStyle(
            "RecStyle", fontSize=22, textColor=white, fontName="Helvetica-Bold",
            alignment=1,
        )
        detail_style = ParagraphStyle(
            "DetailStyle", fontSize=10, textColor=white, fontName="Helvetica",
            alignment=1, leading=16,
        )

        sign = "+" if ret_pct >= 0 else ""
        data = [[
            Paragraph(rec, rec_style),
            Paragraph(
                f"Current: ${current}  →  Target: ${price_target}  ({sign}{ret_pct:.1f}%)<br/>"
                f"Confidence: {confidence}%  |  Composite Score: {composite:.3f} / 1.00",
                detail_style,
            ),
        ]]
        t = Table(data, colWidths=["25%", "75%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), rec_color),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",       (0, 0), (-1, -1), 14),
            ("ROUNDEDCORNERS", [4]),
        ]))
        return t

    def _scorecard_table(self, memo_json: dict) -> Table:
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        HEADER_BG  = HexColor("#1e293b")
        BUY_COLOR  = HexColor("#22c55e")
        HOLD_COLOR = HexColor("#eab308")
        SELL_COLOR = HexColor("#ef4444")

        score_bd = memo_json.get("score_breakdown", {})
        verdicts = memo_json.get("verdicts", {})
        rec      = memo_json.get("recommendation", "HOLD")
        rec_color = BUY_COLOR if rec == "BUY" else (SELL_COLOR if rec == "SELL" else HOLD_COLOR)

        cell_style = ParagraphStyle("sc", fontSize=9, leading=13)
        header_style = ParagraphStyle("sch", fontSize=9, fontName="Helvetica-Bold",
                                       textColor=white, leading=13, alignment=1)

        def cell(text): return Paragraph(str(text), cell_style)
        def hcell(text): return Paragraph(str(text), header_style)

        data = [
            [hcell("Analyst"), hcell("Score"), hcell("Verdict")],
            [cell("Fundamental"), cell(f"{score_bd.get('fundamental_score', 'N/A')}/10"), cell(verdicts.get("fundamental", ""))],
            [cell("Risk"),        cell(f"{score_bd.get('risk_score', 'N/A')}/10"),        cell(verdicts.get("risk", ""))],
            [cell("Sentiment"),   cell(f"{score_bd.get('sentiment_score', 0):+.2f}"),     cell(verdicts.get("sentiment", ""))],
            [Paragraph("<b>COMPOSITE</b>", cell_style),
             Paragraph(f"<b>{memo_json.get('composite_score', 0):.3f}</b>", cell_style),
             Paragraph(f"<b>{rec}</b>", cell_style)],
        ]
        t = Table(data, colWidths=["40%", "25%", "35%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
            ("BACKGROUND",  (0, 4), (-1, 4), rec_color),
            ("TEXTCOLOR",   (0, 4), (-1, 4), white),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN",       (1, 0), (1, -1), "CENTER"),
            ("ALIGN",       (2, 0), (2, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",     (0, 0), (-1, -1), 6),
        ]))
        return t

    def _metrics_table(self, memo_json: dict) -> Table:
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        HEADER_BG = HexColor("#1e293b")
        cell_style = ParagraphStyle("mc", fontSize=9, leading=13)
        hdr_style  = ParagraphStyle("mch", fontSize=9, fontName="Helvetica-Bold",
                                     textColor=white, leading=13)

        kfm = (memo_json.get("memo", {}) or {}).get("key_financial_metrics") or \
              memo_json.get("key_financial_metrics", {})

        rows = [
            [Paragraph("Metric", hdr_style), Paragraph("Value", hdr_style)],
            [Paragraph("Current Price", cell_style),       Paragraph(f"${kfm.get('current_price', 'N/A')}", cell_style)],
            [Paragraph("12M Price Target", cell_style),    Paragraph(f"${kfm.get('price_target_12m', 'N/A')}", cell_style)],
            [Paragraph("Expected Return", cell_style),     Paragraph(f"{kfm.get('upside_downside_pct', 0):+.1f}%", cell_style)],
            [Paragraph("P/E Ratio", cell_style),           Paragraph(f"{kfm.get('pe_ratio', 'N/A')}x", cell_style)],
            [Paragraph("Net Margin", cell_style),          Paragraph(f"{kfm.get('net_margin_pct', 'N/A')}%", cell_style)],
            [Paragraph("Free Cash Flow", cell_style),      Paragraph(f"${kfm.get('fcf_billions', 'N/A')}B", cell_style)],
            [Paragraph("Revenue Growth YoY", cell_style),  Paragraph(f"{kfm.get('revenue_growth_yoy_pct', 'N/A')}%", cell_style)],
        ]
        t = Table(rows, colWidths=["50%", "50%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8fafc"), white]),
            ("ALIGN",       (1, 0), (1, -1), "RIGHT"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",     (0, 0), (-1, -1), 6),
        ]))
        return t

    def _bull_bear_table(self, memo_json: dict, body_style, buy_color, sell_color, light_gray) -> Table:
        from reportlab.lib import colors
        from reportlab.lib.colors import white
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        bull = memo_json.get("bull_case", {})
        bear = memo_json.get("bear_case", {})
        hdr_style  = ParagraphStyle("bbh", fontSize=9, fontName="Helvetica-Bold",
                                     textColor=white, leading=13, alignment=1)
        cell_style = ParagraphStyle("bbc", fontSize=9, leading=13)

        bull_title  = bull.get("title", "Bull Case")
        bear_title  = bear.get("title", "Bear Case")
        bull_thesis = bull.get("thesis", "")
        bear_thesis = bear.get("thesis", "")
        bull_target = bull.get("price_target_bull", "")
        bear_target = bear.get("price_target_bear", "")

        bull_drivers = "\n".join(
            f"• {d.get('driver', '')}: {d.get('detail', '')}"
            for d in bull.get("key_drivers", [])
        )
        bear_risks = "\n".join(
            f"• {r.get('risk', '')}: {r.get('detail', '')}"
            for r in bear.get("key_risks", [])
        )

        data = [
            [Paragraph(f"🟢 {bull_title}", hdr_style), Paragraph(f"🔴 {bear_title}", hdr_style)],
            [Paragraph(bull_thesis, cell_style),         Paragraph(bear_thesis, cell_style)],
            [Paragraph(bull_drivers, cell_style),        Paragraph(bear_risks, cell_style)],
            [Paragraph(f"<b>Bull Target:</b> ${bull_target}", cell_style),
             Paragraph(f"<b>Bear Target:</b> ${bear_target}", cell_style)],
        ]
        t = Table(data, colWidths=["50%", "50%"])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, 0), buy_color),
            ("BACKGROUND",  (1, 0), (1, 0), sell_color),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("PADDING",     (0, 0), (-1, -1), 8),
        ]))
        return t
