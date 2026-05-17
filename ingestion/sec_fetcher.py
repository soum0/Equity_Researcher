import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import Config
from ingestion.utils import rate_limit, retry_with_backoff, sanitize_text, save_json, setup_logger

SECTION_PATTERNS = [
    (r"Item\s+1[Aa]?\.\s*Business", "Business"),
    (r"Item\s+1A\.?\s*Risk\s+Factors", "Risk Factors"),
    (r"Item\s+1B\.?\s*Unresolved\s+Staff\s+Comments", "Unresolved Staff Comments"),
    (r"Item\s+2\.?\s*Properties", "Properties"),
    (r"Item\s+3\.?\s*Legal\s+Proceedings", "Legal Proceedings"),
    (r"Item\s+7\.?\s*Management.s\s+Discussion\s+and\s+Analysis", "MD&A"),
    (r"Management.s\s+Discussion\s+and\s+Analysis", "MD&A"),
    (r"Item\s+7A\.?\s*Quantitative\s+and\s+Qualitative\s+Disclosures", "Quantitative and Qualitative Disclosures"),
    (r"Item\s+8\.?\s*Financial\s+Statements", "Financial Statements"),
    (r"Item\s+9A\.?\s*Controls\s+and\s+Procedures", "Controls and Procedures"),
    (r"Item\s+13\.?\s*Certain\s+Relationships", "Certain Relationships"),
]


class SECFetcher:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker.upper()
        self.config = config
        self.logger = setup_logger("sec_fetcher")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.SEC_USER_AGENT})

    def get_cik(self) -> str:
        self.logger.info(f"[{self.ticker}] Resolving CIK from SEC EDGAR")
        rate_limit(self.config.SEC_DELAY_SECONDS)

        def _fetch():
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(_fetch)

        ticker_lower = self.ticker.lower()
        for entry in data.values():
            if entry.get("ticker", "").lower() == ticker_lower:
                cik = str(entry["cik_str"]).zfill(10)
                self.logger.info(f"[{self.ticker}] CIK resolved: {cik}")
                return cik

        raise ValueError(f"Ticker {self.ticker} not found in SEC company_tickers.json")

    def get_filing_urls(self, cik: str, form_type: str, limit: int) -> list[dict]:
        self.logger.info(f"[{self.ticker}] Fetching {form_type} filing list (CIK={cik})")
        rate_limit(self.config.SEC_DELAY_SECONDS)

        def _fetch():
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(_fetch)
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        results = []
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            raw_acc = accessions[i]
            acc_nodash = raw_acc.replace("-", "")
            cik_nodash = cik.lstrip("0")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_nodash}"
                f"/{acc_nodash}/{primary_docs[i]}"
            )
            results.append({
                "accession_number": raw_acc,
                "filing_date": dates[i],
                "form_type": form,
                "primary_document": primary_docs[i],
                "filing_url": filing_url,
                "report_date": report_dates[i] if i < len(report_dates) else None,
                "cik_nodash": cik_nodash,
                "acc_nodash": acc_nodash,
            })
            if len(results) >= limit:
                break

        self.logger.info(f"[{self.ticker}] Found {len(results)} {form_type} filings")
        return results

    def download_filing_text(self, filing_meta: dict) -> str:
        url = filing_meta["filing_url"]
        self.logger.info(f"[{self.ticker}] Downloading filing: {url}")
        rate_limit(self.config.SEC_DELAY_SECONDS)

        def _fetch_url(target_url: str) -> str:
            resp = self.session.get(target_url, timeout=60, stream=True)
            resp.raise_for_status()

            chunks = []
            for chunk in resp.iter_content(chunk_size=1024 * 1024, decode_unicode=False):
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)

            soup = BeautifulSoup(raw_bytes, "lxml")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            return sanitize_text(text)

        try:
            text = _fetch_url(url)
            if len(text) > 5000:
                return text
        except Exception as exc:
            self.logger.warning(f"[{self.ticker}] Primary doc failed ({exc}), trying index")

        rate_limit(self.config.SEC_DELAY_SECONDS)
        cik_nodash = filing_meta["cik_nodash"]
        acc_nodash = filing_meta["acc_nodash"]
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_nodash}"
            f"/{acc_nodash}/{filing_meta['accession_number']}-index.htm"
        )

        def _fetch_index():
            resp = self.session.get(index_url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")
            links = soup.find_all("a", href=True)
            htm_links = [
                a["href"] for a in links
                if a["href"].lower().endswith((".htm", ".html"))
                and "ix?doc=" not in a["href"].lower()
            ]
            return htm_links

        try:
            htm_links = _fetch_index()
            if htm_links:
                best = max(htm_links, key=lambda h: len(h))
                base = f"https://www.sec.gov"
                full_url = best if best.startswith("http") else base + best
                rate_limit(self.config.SEC_DELAY_SECONDS)
                return _fetch_url(full_url)
        except Exception as exc:
            self.logger.error(f"[{self.ticker}] Index fallback also failed: {exc}")

        return ""

    def detect_sections(self, text: str) -> list[str]:
        found = []
        for pattern, label in SECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if label not in found:
                    found.append(label)
        return found

    def fetch_all(self) -> dict:
        cik = None
        try:
            cik = self.get_cik()
        except Exception as exc:
            self.logger.error(f"[{self.ticker}] CIK lookup failed: {exc}")
            return {"ticker": self.ticker, "cik": None, "filings": [], "error": str(exc)}

        sec_dir = Path(f"{self.config.DATA_DIR}/{self.ticker}/sec_filings")
        sec_dir.mkdir(parents=True, exist_ok=True)

        filing_records = []

        for form_type in ["10-K", "10-Q"]:
            try:
                filings = self.get_filing_urls(cik, form_type, self.config.MAX_SEC_FILINGS)
            except Exception as exc:
                self.logger.error(f"[{self.ticker}] get_filing_urls({form_type}) failed: {exc}")
                continue

            for meta in filings:
                try:
                    text = self.download_filing_text(meta)
                except Exception as exc:
                    self.logger.error(f"[{self.ticker}] Download failed for {meta['accession_number']}: {exc}")
                    text = ""

                filename = f"{form_type.replace('-', '_')}_{meta['filing_date']}.txt"
                filepath = sec_dir / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)

                sections = self.detect_sections(text)
                self.logger.info(
                    f"[{self.ticker}] Saved {filename}: {len(text):,} chars, sections={sections}"
                )

                filing_records.append({
                    "form_type": form_type,
                    "filing_date": meta["filing_date"],
                    "fiscal_year_end": meta.get("report_date"),
                    "accession_number": meta["accession_number"],
                    "filename": filename,
                    "char_count": len(text),
                    "sections_detected": sections,
                })

        metadata = {
            "ticker": self.ticker,
            "cik": cik,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filings": filing_records,
        }

        save_json(metadata, str(sec_dir / "metadata.json"))
        self.logger.info(f"[{self.ticker}] SEC fetch complete: {len(filing_records)} filings saved")
        return metadata
