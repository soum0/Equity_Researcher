from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime

import tiktoken

from config import Config


class SECChunker:
    SECTION_PATTERNS = {
        "Business Overview": r"Item\s+1\.?\s+Business",
        "Risk Factors": r"Item\s+1A\.?\s+Risk\s+Factors",
        "Legal Proceedings": r"Item\s+3\.?\s+Legal\s+Proceedings",
        "MD&A": r"Item\s+7\.?\s+Management.s\s+Discussion",
        "Quantitative Risk": r"Item\s+7A\.?\s+Quantitative",
        "Financial Statements": r"Item\s+8\.?\s+Financial\s+Statements",
        "Risk Controls": r"Item\s+9A\.?\s+Controls",
        "Executive Compensation": r"Item\s+11\.?\s+Executive\s+Compensation",
    }

    def __init__(self, config: Config):
        self.config = config
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _find_readable_start(self, text: str) -> int:
        """Return the character offset where the human-readable filing text begins."""
        for marker in ("This Annual Report on Form", "This Quarterly Report on Form",
                       "UNITED STATES\nSECURITIES", "PART I\n", "Part I\n"):
            idx = text.find(marker)
            if idx != -1:
                return idx
        # Fallback: skip the XBRL header by finding the first long English sentence
        # (at least 80 printable ASCII chars with spaces)
        for match in re.finditer(r'[A-Za-z][A-Za-z ,.\-\'"()\d;:]{80,}', text):
            return match.start()
        return 0

    def split_into_sections(self, text: str) -> list[dict]:
        start = self._find_readable_start(text)
        readable = text[start:]

        boundaries: list[tuple[int, str]] = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            for m in re.finditer(pattern, readable, re.IGNORECASE):
                boundaries.append((m.start(), section_name))

        if not boundaries:
            return [{"section_name": "Full Document", "text": readable,
                     "char_start": start, "char_end": start + len(readable)}]

        boundaries.sort(key=lambda x: x[0])

        sections: list[dict] = []
        for i, (pos, name) in enumerate(boundaries):
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(readable)
            section_text = readable[pos:end_pos].strip()
            if section_text:
                sections.append({
                    "section_name": name,
                    "text": section_text,
                    "char_start": start + pos,
                    "char_end": start + end_pos,
                })

        return sections

    def chunk_section(self, section: dict, doc_meta: dict) -> list[dict]:
        text = section["text"]
        words = text.split()
        tokens = self._enc.encode(text)

        chunk_size = self.config.CHUNK_SIZE_TOKENS
        overlap = self.config.CHUNK_OVERLAP_TOKENS
        min_tokens = self.config.MIN_CHUNK_TOKENS

        chunks: list[dict] = []
        start_tok = 0

        while start_tok < len(tokens):
            end_tok = min(start_tok + chunk_size, len(tokens))
            chunk_tokens = tokens[start_tok:end_tok]
            chunk_text = self._enc.decode(chunk_tokens)
            count = len(chunk_tokens)

            if count >= min_tokens:
                idx = len(chunks) + 1
                ticker = doc_meta.get("ticker", "UNKN")
                form = doc_meta.get("form_type", "DOC").replace("-", "").replace("/", "")
                date = doc_meta.get("filing_date", "")[:10].replace("-", "")
                section_slug = re.sub(r"[^A-Za-z0-9]", "", section["section_name"])
                base = f"{ticker}_{form}_{date}_{section_slug}_{idx:03d}"
                hash_suffix = hashlib.md5(chunk_text[:50].encode()).hexdigest()[:6]
                chunk_id = f"{base}_{hash_suffix}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "ticker": ticker,
                    "source_file": doc_meta.get("source_file", ""),
                    "form_type": doc_meta.get("form_type", ""),
                    "filing_date": doc_meta.get("filing_date", ""),
                    "section": section["section_name"],
                    "chunk_index": idx,
                    "total_chunks_in_section": 0,  # filled in below
                    "text": chunk_text,
                    "token_count": count,
                    "char_start": section["char_start"],
                    "char_end": section["char_end"],
                })

            if end_tok >= len(tokens):
                break
            start_tok = end_tok - overlap

        total = len(chunks)
        for c in chunks:
            c["total_chunks_in_section"] = total

        return chunks

    def chunk_news_articles(self, news_data: dict) -> list[dict]:
        articles = news_data.get("articles", [])
        ticker = news_data.get("ticker", "UNKN")
        chunks: list[dict] = []

        for i, article in enumerate(articles, start=1):
            pub = article.get("published_at", "")[:10].replace("-", "")
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline}. {summary}".strip()
            if not text or text == ".":
                continue

            count = self._token_count(text)
            base = f"{ticker}_NEWS_{pub}_{i:03d}"
            hash_suffix = hashlib.md5(text[:50].encode()).hexdigest()[:6]
            chunk_id = f"{base}_{hash_suffix}"
            chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "source_file": "news_articles.json",
                "form_type": "NEWS",
                "filing_date": article.get("published_at", ""),
                "section": "News",
                "chunk_index": i,
                "text": text,
                "token_count": count,
                "sentiment_score": article.get("sentiment_score"),
                "url": article.get("url", ""),
            })

        return chunks

    def chunk_financials(self, company_data: dict, key_metrics: dict) -> list[dict]:
        ticker = company_data.get("ticker", "UNKN")
        name = company_data.get("name", ticker)
        chunks: list[dict] = []

        basic = key_metrics.get("basic_financials", {})
        per_share = basic.get("per_share", {})
        valuation = basic.get("valuation", {})
        margins = basic.get("margins", {})
        growth = basic.get("growth", {})
        momentum = basic.get("momentum", {})

        metric_groups = {
            "valuation": (
                f"{name} valuation metrics: P/E (TTM) {valuation.get('pe_ttm', 'N/A')}x, "
                f"P/B {valuation.get('pb_annual', 'N/A')}x, "
                f"EV/EBITDA {valuation.get('ev_ebitda', 'N/A')}x. "
                f"EPS (TTM) ${per_share.get('eps_ttm', 'N/A')}, "
                f"revenue per share (TTM) ${per_share.get('revenue_per_share_ttm', 'N/A')}."
            ),
            "profitability": (
                f"{name} profitability: gross margin {margins.get('gross_margin_ttm', 'N/A')}%, "
                f"operating margin {margins.get('operating_margin_ttm', 'N/A')}%, "
                f"net margin {margins.get('net_margin_ttm', 'N/A')}%."
            ),
            "growth": (
                f"{name} growth metrics: revenue growth TTM {growth.get('revenue_growth_ttm', 'N/A')}%, "
                f"3-year revenue growth {growth.get('revenue_growth_3y', 'N/A')}%, "
                f"3-year EPS growth {growth.get('eps_growth_3y', 'N/A')}%."
            ),
            "momentum": (
                f"{name} price momentum: 52-week high ${momentum.get('52w_high', 'N/A')}, "
                f"52-week low ${momentum.get('52w_low', 'N/A')}, "
                f"52-week return {momentum.get('52w_return', 'N/A')}%, "
                f"beta {momentum.get('beta', 'N/A')}."
            ),
        }

        analyst = key_metrics.get("analyst_recommendations", {})
        if analyst.get("consensus"):
            metric_groups["analyst"] = (
                f"{name} analyst consensus: {analyst['consensus']} "
                f"(latest period {analyst.get('latest_period', 'N/A')}). "
                f"Breakdown: strong buy {analyst.get('breakdown', [{}])[0].get('strong_buy', 'N/A')}, "
                f"buy {analyst.get('breakdown', [{}])[0].get('buy', 'N/A')}, "
                f"hold {analyst.get('breakdown', [{}])[0].get('hold', 'N/A')}, "
                f"sell {analyst.get('breakdown', [{}])[0].get('sell', 'N/A')}."
            )

        today = datetime.utcnow().strftime("%Y%m%d")
        for i, (group, text) in enumerate(metric_groups.items(), start=1):
            base = f"{ticker}_FINANCIALS_{today}_{group.upper()}_{i:03d}"
            hash_suffix = hashlib.md5(text[:50].encode()).hexdigest()[:6]
            chunk_id = f"{base}_{hash_suffix}"
            chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "source_file": "key_metrics.json",
                "form_type": "FINANCIALS",
                "filing_date": basic.get("as_of_date", ""),
                "section": f"Financials_{group}",
                "chunk_index": i,
                "text": text,
                "token_count": self._token_count(text),
            })

        return chunks

    def chunk_macro(self, macro_data: dict) -> list[dict]:
        indicators = macro_data.get("indicators", {})
        parts: list[str] = ["Current macroeconomic conditions:"]

        label_map = {
            "federal_funds_rate": "Federal Funds Rate",
            "cpi_inflation": "CPI inflation",
            "gdp_growth": "GDP growth",
            "unemployment_rate": "unemployment rate",
            "10y_treasury_yield": "10Y Treasury yield",
            "vix_index": "VIX",
        }

        for key, ind in indicators.items():
            if "error" in ind:
                continue
            label = label_map.get(key, key.replace("_", " ").title())
            val = ind.get("latest_value", "N/A")
            unit = ind.get("unit", "")
            trend = ind.get("trend_3m", "")
            trend_str = f" ({trend} trend)" if trend else ""
            parts.append(f"{label} at {val}{unit}{trend_str}")

        text = ", ".join(parts[1:])
        full_text = parts[0] + " " + text + "."
        base = f"MACRO_{datetime.utcnow().strftime('%Y%m%d')}_001"
        hash_suffix = hashlib.md5(full_text[:50].encode()).hexdigest()[:6]
        chunk_id = f"{base}_{hash_suffix}"

        return [{
            "chunk_id": chunk_id,
            "ticker": "MACRO",
            "source_file": "macro_indicators.json",
            "form_type": "MACRO",
            "filing_date": macro_data.get("fetched_at", "")[:10],
            "section": "Macro",
            "chunk_index": 1,
            "text": full_text,
            "token_count": self._token_count(full_text),
        }]

    def process_all(self, ticker: str) -> list[dict]:
        data_dir = f"{self.config.DATA_DIR}/{ticker}"
        graphrag_dir = f"{data_dir}/graphrag"
        os.makedirs(graphrag_dir, exist_ok=True)

        all_chunks: list[dict] = []
        sec_count = 0
        section_counts: dict[str, int] = {}

        # SEC filings
        if self.config.INDEX_SEC_FILINGS:
            filings_dir = f"{data_dir}/sec_filings"
            meta_path = f"{filings_dir}/metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
                for filing in metadata.get("filings", []):
                    fpath = f"{filings_dir}/{filing['filename']}"
                    if not os.path.exists(fpath):
                        continue
                    with open(fpath) as f:
                        text = f.read()
                    doc_meta = {
                        "ticker": ticker,
                        "form_type": filing["form_type"],
                        "filing_date": filing["filing_date"],
                        "source_file": filing["filename"],
                    }
                    sections = self.split_into_sections(text)
                    for sec in sections:
                        chunks = self.chunk_section(sec, doc_meta)
                        all_chunks.extend(chunks)
                        sec_count += len(chunks)
                        section_counts[sec["section_name"]] = (
                            section_counts.get(sec["section_name"], 0) + len(chunks)
                        )

        # News articles
        news_chunks: list[dict] = []
        if self.config.INDEX_NEWS:
            news_path = f"{data_dir}/news_articles.json"
            if os.path.exists(news_path):
                with open(news_path) as f:
                    news_data = json.load(f)
                news_chunks = self.chunk_news_articles(news_data)
                all_chunks.extend(news_chunks)

        # Financials
        fin_chunks: list[dict] = []
        if self.config.INDEX_FINANCIALS:
            company_path = f"{data_dir}/company_profile.json"
            metrics_path = f"{data_dir}/key_metrics.json"
            if os.path.exists(company_path) and os.path.exists(metrics_path):
                with open(company_path) as f:
                    company_data = json.load(f)
                with open(metrics_path) as f:
                    key_metrics = json.load(f)
                fin_chunks = self.chunk_financials(company_data, key_metrics)
                all_chunks.extend(fin_chunks)

        # Macro
        macro_chunks: list[dict] = []
        if self.config.INDEX_MACRO:
            macro_path = f"{data_dir}/macro_indicators.json"
            if os.path.exists(macro_path):
                with open(macro_path) as f:
                    macro_data = json.load(f)
                macro_chunks = self.chunk_macro(macro_data)
                all_chunks.extend(macro_chunks)

        # Reassign global sequential IDs
        for i, chunk in enumerate(all_chunks, start=1):
            chunk["global_index"] = i

        # Save
        out_path = f"{graphrag_dir}/chunks.json"
        with open(out_path, "w") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"      SEC filings:   {sec_count} chunks across {len(section_counts)} sections")
        print(f"      News articles: {len(news_chunks)} chunks")
        print(f"      Financials:    {len(fin_chunks)} chunks")
        print(f"      Macro:         {len(macro_chunks)} chunks")
        print(f"      Total:         {len(all_chunks)} chunks  OK")

        return all_chunks
