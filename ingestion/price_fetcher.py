from datetime import date, datetime
from typing import Any

import finnhub
import yfinance as yf

from config import Config
from ingestion.utils import rate_limit, retry_with_backoff, save_json, setup_logger


class PriceFetcher:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker.upper()
        self.config = config
        self.logger = setup_logger("price_fetcher")
        self.finnhub_client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    def fetch_price_history(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching price history ({self.config.PRICE_HISTORY_PERIOD})")

        def _fetch():
            yf_ticker = yf.Ticker(self.ticker)
            hist = yf_ticker.history(period=self.config.PRICE_HISTORY_PERIOD)
            return yf_ticker, hist

        yf_ticker, hist = retry_with_backoff(_fetch)

        if hist.empty:
            self.logger.warning(f"[{self.ticker}] No price history returned")
            return {"ticker": self.ticker, "period": self.config.PRICE_HISTORY_PERIOD, "history": [], "summary": {}}

        hist = hist.reset_index()
        records = []
        for _, row in hist.iterrows():
            dt = row["Date"]
            if hasattr(dt, "date"):
                dt = dt.date()
            records.append({
                "date": str(dt),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
                "dividends": round(float(row.get("Dividends", 0.0)), 6),
                "stock_splits": round(float(row.get("Stock Splits", 0.0)), 4),
            })

        closes = [r["close"] for r in records]
        volumes = [r["volume"] for r in records]

        latest_close = closes[-1] if closes else None
        high_52w = max(closes[-252:]) if len(closes) >= 252 else max(closes)
        low_52w = min(closes[-252:]) if len(closes) >= 252 else min(closes)
        avg_vol_30d = int(sum(volumes[-30:]) / min(30, len(volumes)))

        def pct_change(n_days: int) -> float | None:
            if len(closes) < n_days + 1:
                return None
            old = closes[-(n_days + 1)]
            return round((closes[-1] - old) / old * 100, 2) if old else None

        info = {}
        try:
            info = yf_ticker.info or {}
        except Exception:
            pass

        result = {
            "ticker": self.ticker,
            "period": self.config.PRICE_HISTORY_PERIOD,
            "currency": info.get("currency", "USD"),
            "data_points": len(records),
            "history": records,
            "summary": {
                "latest_close": latest_close,
                "52w_high": high_52w,
                "52w_low": low_52w,
                "avg_volume_30d": avg_vol_30d,
                "price_change_1m_pct": pct_change(21),
                "price_change_3m_pct": pct_change(63),
                "price_change_1y_pct": pct_change(252),
            },
        }
        self.logger.info(f"[{self.ticker}] Price history: {len(records)} data points")
        return result

    def fetch_key_ratios(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching key ratios from yfinance")

        def _fetch():
            return yf.Ticker(self.ticker).info or {}

        info = retry_with_backoff(_fetch)

        def g(key: str) -> Any:
            val = info.get(key)
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return val

        ex_div = info.get("exDividendDate")
        if ex_div and not isinstance(ex_div, str):
            try:
                ex_div = date.fromtimestamp(int(ex_div)).isoformat()
            except Exception:
                ex_div = str(ex_div)

        result = {
            "ticker": self.ticker,
            "as_of_date": date.today().isoformat(),
            "valuation": {
                "market_cap": g("marketCap"),
                "enterprise_value": g("enterpriseValue"),
                "pe_ratio_ttm": g("trailingPE"),
                "forward_pe": g("forwardPE"),
                "pb_ratio": g("priceToBook"),
                "ps_ratio": g("priceToSalesTrailing12Months"),
                "ev_to_ebitda": g("enterpriseToEbitda"),
                "peg_ratio": g("pegRatio"),
            },
            "profitability": {
                "gross_margin": g("grossMargins"),
                "operating_margin": g("operatingMargins"),
                "net_margin": g("profitMargins"),
                "roe": g("returnOnEquity"),
                "roa": g("returnOnAssets"),
                "roic": None,
            },
            "growth": {
                "revenue_growth_yoy": g("revenueGrowth"),
                "earnings_growth_yoy": g("earningsGrowth"),
                "earnings_quarterly_growth": g("earningsQuarterlyGrowth"),
            },
            "financial_health": {
                "current_ratio": g("currentRatio"),
                "quick_ratio": g("quickRatio"),
                "debt_to_equity": g("debtToEquity"),
                "total_debt": g("totalDebt"),
                "free_cash_flow": g("freeCashflow"),
                "cash_and_equivalents": g("totalCash"),
            },
            "dividend": {
                "dividend_yield": g("dividendYield"),
                "dividend_rate": g("dividendRate"),
                "payout_ratio": g("payoutRatio"),
                "ex_dividend_date": ex_div,
            },
        }
        self.logger.info(f"[{self.ticker}] Key ratios fetched")
        return result

    def fetch_realtime_quote(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching real-time quote from Finnhub")
        rate_limit(self.config.FINNHUB_DELAY_SECONDS)

        def _fetch():
            return self.finnhub_client.quote(self.ticker)

        q = retry_with_backoff(_fetch)

        ts = q.get("t")
        if ts:
            try:
                ts = datetime.utcfromtimestamp(int(ts)).isoformat() + "Z"
            except Exception:
                ts = str(ts)

        result = {
            "current": q.get("c"),
            "open": q.get("o"),
            "high": q.get("h"),
            "low": q.get("l"),
            "prev_close": q.get("pc"),
            "change": q.get("d"),
            "change_pct": q.get("dp"),
            "timestamp": ts,
        }
        self.logger.info(f"[{self.ticker}] Real-time quote: ${result['current']}")
        return result

    def fetch_all(self) -> dict:
        combined = {}
        errors = []

        try:
            combined["price_history"] = self.fetch_price_history()
        except Exception as exc:
            errors.append(f"price_history: {exc}")
            self.logger.error(f"[{self.ticker}] price_history failed: {exc}")
            combined["price_history"] = {}

        try:
            combined["key_ratios"] = self.fetch_key_ratios()
        except Exception as exc:
            errors.append(f"key_ratios: {exc}")
            self.logger.error(f"[{self.ticker}] key_ratios failed: {exc}")
            combined["key_ratios"] = {}

        try:
            combined["realtime_quote"] = self.fetch_realtime_quote()
        except Exception as exc:
            errors.append(f"realtime_quote: {exc}")
            self.logger.error(f"[{self.ticker}] realtime_quote failed: {exc}")
            combined["realtime_quote"] = {}

        combined["errors"] = errors

        filepath = f"{self.config.DATA_DIR}/{self.ticker}/price_data.json"
        save_json(combined, filepath)
        self.logger.info(f"[{self.ticker}] Price data saved to {filepath}")
        return combined
