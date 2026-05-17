from datetime import datetime, timezone
from pathlib import Path

from config import Config
from ingestion.company_fetcher import CompanyFetcher
from ingestion.macro_fetcher import MacroFetcher
from ingestion.news_fetcher import NewsFetcher
from ingestion.price_fetcher import PriceFetcher
from ingestion.sec_fetcher import SECFetcher
from ingestion.utils import ensure_dir, save_json, setup_logger


class DataIngestionPipeline:
    def __init__(self, ticker: str, config: Config = None):
        self.ticker = ticker.upper()
        self.config = config or Config()
        self.logger = setup_logger("pipeline")

    def run(self) -> dict:
        ticker = self.ticker
        config = self.config

        ensure_dir(f"{config.DATA_DIR}/{ticker}")

        results: dict = {}
        errors: list[str] = []

        # ── 1. Price data ──────────────────────────────────────────────────
        print(f"[1/5] Fetching price data...")
        try:
            price_data = PriceFetcher(ticker, config).fetch_all()
            results["price"] = price_data
            n_days = price_data.get("price_history", {}).get("data_points", 0)
            latest = price_data.get("price_history", {}).get("summary", {}).get("latest_close", "?")
            print(f"[1/5] Price data:        OK — {n_days} days, latest ${latest}")
        except Exception as exc:
            errors.append(f"price: {exc}")
            results["price"] = {}
            print(f"[1/5] Price data:        FAILED — {exc}")
            self.logger.error(f"[{ticker}] PriceFetcher failed: {exc}")

        # ── 2. Company profile ─────────────────────────────────────────────
        print(f"[2/5] Fetching company profile...")
        company_name = ticker
        try:
            company_data = CompanyFetcher(ticker, config).fetch_all()
            results["company"] = company_data
            company_name = company_data.get("name") or ticker
            exchange = company_data.get("exchange", "")
            print(f"[2/5] Company profile:   OK — {company_name} ({exchange})")
        except Exception as exc:
            errors.append(f"company: {exc}")
            results["company"] = {}
            print(f"[2/5] Company profile:   FAILED — {exc}")
            self.logger.error(f"[{ticker}] CompanyFetcher failed: {exc}")

        # ── 3. News articles ───────────────────────────────────────────────
        print(f"[3/5] Fetching news articles...")
        try:
            news_data = NewsFetcher(ticker, company_name, config).fetch_all()
            results["news"] = news_data
            n_articles = news_data.get("total_articles", 0)
            print(f"[3/5] News articles:     OK — {n_articles} articles ({config.NEWS_LOOKBACK_DAYS} days)")
        except Exception as exc:
            errors.append(f"news: {exc}")
            results["news"] = {}
            print(f"[3/5] News articles:     FAILED — {exc}")
            self.logger.error(f"[{ticker}] NewsFetcher failed: {exc}")

        # ── 4. SEC filings ─────────────────────────────────────────────────
        print(f"[4/5] Fetching SEC filings...")
        try:
            sec_data = SECFetcher(ticker, config).fetch_all()
            results["sec"] = sec_data
            n_filings = len(sec_data.get("filings", []))
            total_chars = sum(f.get("char_count", 0) for f in sec_data.get("filings", []))
            nk = sum(1 for f in sec_data.get("filings", []) if f["form_type"] == "10-K")
            nq = sum(1 for f in sec_data.get("filings", []) if f["form_type"] == "10-Q")
            print(f"[4/5] SEC filings:       OK — {n_filings} filings ({nk}×10-K, {nq}×10-Q, {total_chars:,} chars)")
        except Exception as exc:
            errors.append(f"sec: {exc}")
            results["sec"] = {}
            print(f"[4/5] SEC filings:       FAILED — {exc}")
            self.logger.error(f"[{ticker}] SECFetcher failed: {exc}")

        # ── 5. Macro indicators ────────────────────────────────────────────
        print(f"[5/5] Fetching macro indicators...")
        try:
            macro_data = MacroFetcher(config).fetch_all(ticker=ticker)
            results["macro"] = macro_data
            n_series = len([k for k, v in macro_data.get("indicators", {}).items() if "error" not in v])
            print(f"[5/5] Macro indicators:  OK — {n_series} FRED series")
        except Exception as exc:
            errors.append(f"macro: {exc}")
            results["macro"] = {}
            print(f"[5/5] Macro indicators:  FAILED — {exc}")
            self.logger.error(f"[{ticker}] MacroFetcher failed: {exc}")

        # ── Generate and save report ───────────────────────────────────────
        report = self.generate_report(results, errors, company_name)
        save_json(report, f"{config.DATA_DIR}/{ticker}/ingestion_report.json")

        self._print_summary(report)
        return report

    def generate_report(self, results: dict, errors: list[str], company_name: str) -> dict:
        ticker = self.ticker
        config = self.config

        price_history = results.get("price", {}).get("price_history", {})
        news = results.get("news", {})
        sec = results.get("sec", {})
        macro = results.get("macro", {})

        n_filings = len(sec.get("filings", []))
        total_chars = sum(f.get("char_count", 0) for f in sec.get("filings", []))
        n_macro = len([k for k, v in macro.get("indicators", {}).items() if "error" not in v])

        fetcher_results = {}
        for key, filename in [
            ("price", "price_data.json"),
            ("company", "company_profile.json"),
            ("news", "news_articles.json"),
            ("sec", "sec_filings/metadata.json"),
            ("macro", "macro_indicators.json"),
        ]:
            if results.get(key):
                fetcher_results[key] = {"status": "ok", "file": filename}
            else:
                fetcher_results[key] = {"status": "failed", "file": filename}

        n_ok = sum(1 for v in fetcher_results.values() if v["status"] == "ok")
        if n_ok == 5:
            status = "complete"
        elif n_ok == 0:
            status = "failed"
        else:
            status = "partial"

        return {
            "ticker": ticker,
            "ingestion_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "data_dir": f"{config.DATA_DIR}/{ticker}/",
            "summary": {
                "price_history_days": price_history.get("data_points", 0),
                "latest_price": price_history.get("summary", {}).get("latest_close"),
                "company_name": company_name,
                "news_articles_fetched": news.get("total_articles", 0),
                "sec_filings_fetched": n_filings,
                "sec_total_chars": total_chars,
                "macro_indicators_fetched": n_macro,
            },
            "fetcher_results": fetcher_results,
            "errors": errors,
        }

    def _print_summary(self, report: dict):
        s = report.get("summary", {})
        ticker = report["ticker"]
        border = "=" * 50
        print(f"\n{border}")
        status_icon = "OK" if report["status"] == "complete" else "PARTIAL" if report["status"] == "partial" else "FAILED"
        print(f"[{status_icon}] Ingestion {report['status'].upper()} — {ticker}")
        print(f"   Company:        {s.get('company_name', 'N/A')}")
        price = s.get("latest_price")
        print(f"   Latest Price:   ${price:.2f}" if price else "   Latest Price:   N/A")
        print(f"   News Articles:  {s.get('news_articles_fetched', 0)}")
        chars = s.get("sec_total_chars", 0)
        print(f"   SEC Filings:    {s.get('sec_filings_fetched', 0)} ({chars:,} chars)")
        print(f"   Macro Series:   {s.get('macro_indicators_fetched', 0)}")
        print(f"   Output Dir:     {report.get('data_dir', '')}")
        print(f"   Report:         {report.get('data_dir', '')}ingestion_report.json")
        print(f"{border}\n")
        if report.get("errors"):
            print("Errors encountered:")
            for e in report["errors"]:
                print(f"  - {e}")
