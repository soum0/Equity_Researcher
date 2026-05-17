from datetime import date, datetime, timedelta

import finnhub

from config import Config
from ingestion.utils import rate_limit, retry_with_backoff, save_json, setup_logger


class CompanyFetcher:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker.upper()
        self.config = config
        self.logger = setup_logger("company_fetcher")
        self.client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    def _rl(self):
        rate_limit(self.config.FINNHUB_DELAY_SECONDS)

    def fetch_company_profile(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching company profile from Finnhub")
        self._rl()

        def _fetch():
            return self.client.company_profile2(symbol=self.ticker)

        raw = retry_with_backoff(_fetch) or {}

        result = {
            "name": raw.get("name"),
            "ticker": self.ticker,
            "exchange": raw.get("exchange"),
            "industry": raw.get("finnhubIndustry"),
            "sector": raw.get("gics"),
            "country": raw.get("country"),
            "currency": raw.get("currency"),
            "market_cap": raw.get("marketCapitalization"),
            "shares_outstanding": raw.get("shareOutstanding"),
            "website": raw.get("weburl"),
            "logo": raw.get("logo"),
            "ipo_date": raw.get("ipo"),
            "description": raw.get("name"),
        }
        self.logger.info(f"[{self.ticker}] Profile: {result.get('name')} ({result.get('exchange')})")
        return result

    def fetch_basic_financials(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching basic financials from Finnhub")
        self._rl()

        def _fetch():
            return self.client.company_basic_financials(self.ticker, "all")

        raw = retry_with_backoff(_fetch) or {}
        metric = raw.get("metric", {})

        result = {
            "as_of_date": date.today().isoformat(),
            "per_share": {
                "eps_ttm": metric.get("epsTTM"),
                "book_value_per_share": metric.get("bookValuePerShareAnnual"),
                "cash_per_share": metric.get("cashPerSharePerShareAnnual"),
                "revenue_per_share_ttm": metric.get("revenuePerShareTTM"),
            },
            "valuation": {
                "pe_ttm": metric.get("peTTM"),
                "pb_annual": metric.get("pbAnnual"),
                "ev_ebitda": metric.get("currentEv/freeCashFlowTTM"),
            },
            "margins": {
                "gross_margin_ttm": metric.get("grossMarginTTM"),
                "net_margin_ttm": metric.get("netProfitMarginTTM"),
                "operating_margin_ttm": metric.get("operatingMarginTTM"),
            },
            "growth": {
                "revenue_growth_3y": metric.get("revenueGrowth3Y"),
                "eps_growth_3y": metric.get("epsGrowth3Y"),
                "revenue_growth_ttm": metric.get("revenueGrowthTTMYoy"),
            },
            "momentum": {
                "52w_high": metric.get("52WeekHigh"),
                "52w_low": metric.get("52WeekLow"),
                "52w_return": metric.get("52WeekPriceReturnDaily"),
                "beta": metric.get("beta"),
                "rsi14": metric.get("rsi14"),
            },
        }
        self.logger.info(f"[{self.ticker}] Basic financials fetched")
        return result

    def fetch_analyst_recommendations(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching analyst recommendations from Finnhub")
        self._rl()

        def _fetch():
            return self.client.recommendation_trends(self.ticker)

        raw = retry_with_backoff(_fetch) or []

        breakdown = []
        for item in raw[:4]:
            total = sum([
                item.get("strongBuy", 0),
                item.get("buy", 0),
                item.get("hold", 0),
                item.get("sell", 0),
                item.get("strongSell", 0),
            ])
            sb = item.get("strongBuy", 0)
            b = item.get("buy", 0)
            h = item.get("hold", 0)
            s = item.get("sell", 0)
            ss = item.get("strongSell", 0)
            score = round((sb * 5 + b * 4 + h * 3 + s * 2 + ss * 1) / total, 2) if total else None
            breakdown.append({
                "period": item.get("period"),
                "strong_buy": sb,
                "buy": b,
                "hold": h,
                "sell": s,
                "strong_sell": ss,
                "total": total,
                "consensus_score": score,
            })

        consensus = "HOLD"
        if breakdown:
            latest = breakdown[0]
            buys = latest["strong_buy"] + latest["buy"]
            sells = latest["sell"] + latest["strong_sell"]
            if buys > sells * 2:
                consensus = "BUY"
            elif sells > buys * 2:
                consensus = "SELL"
            elif buys > sells:
                consensus = "OVERWEIGHT"

        result = {
            "latest_period": breakdown[0]["period"] if breakdown else None,
            "consensus": consensus,
            "breakdown": breakdown,
        }
        self.logger.info(f"[{self.ticker}] Analyst consensus: {consensus}")
        return result

    def fetch_earnings_history(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching earnings history from Finnhub")
        self._rl()

        def _fetch():
            return self.client.company_earnings(self.ticker, limit=8)

        raw = retry_with_backoff(_fetch) or []

        quarters = []
        beats = 0
        surprise_pcts = []
        for item in raw:
            actual = item.get("actual")
            estimate = item.get("estimate")
            surprise = None
            surprise_pct = None
            if actual is not None and estimate is not None and estimate != 0:
                surprise = round(actual - estimate, 4)
                surprise_pct = round((surprise / abs(estimate)) * 100, 2)
                if actual > estimate:
                    beats += 1
                surprise_pcts.append(surprise_pct)
            quarters.append({
                "period": item.get("period"),
                "actual_eps": actual,
                "estimated_eps": estimate,
                "surprise": surprise,
                "surprise_pct": surprise_pct,
            })

        beat_rate = round(beats / len(quarters), 3) if quarters else None
        avg_surprise = round(sum(surprise_pcts) / len(surprise_pcts), 2) if surprise_pcts else None

        result = {
            "quarters": quarters,
            "beat_rate": beat_rate,
            "avg_surprise_pct": avg_surprise,
        }
        self.logger.info(f"[{self.ticker}] Earnings: {len(quarters)} quarters, beat rate {beat_rate}")
        return result

    def fetch_insider_transactions(self) -> dict:
        self.logger.info(f"[{self.ticker}] Fetching insider transactions from Finnhub")
        self._rl()

        def _fetch():
            return self.client.stock_insider_transactions(self.ticker)

        raw = retry_with_backoff(_fetch) or {}
        all_txns = raw.get("data", [])

        cutoff = (datetime.utcnow() - timedelta(days=90)).date()
        filtered = []
        for txn in all_txns:
            txn_date_str = txn.get("transactionDate", "")
            try:
                txn_date = date.fromisoformat(txn_date_str)
            except Exception:
                continue
            if txn_date < cutoff:
                continue

            shares = txn.get("share", 0) or 0
            price = txn.get("transactionPrice", 0) or 0
            change = txn.get("change", 0) or 0
            txn_type = "BUY" if change > 0 else "SELL"
            filtered.append({
                "name": txn.get("name"),
                "title": txn.get("filingForm", ""),
                "transaction_type": txn_type,
                "shares": abs(int(shares)),
                "value_usd": round(abs(shares) * price, 2) if price else None,
                "date": txn_date_str,
            })

        net_shares = sum(
            t["shares"] if t["transaction_type"] == "BUY" else -t["shares"]
            for t in filtered
        )

        if net_shares > 50000:
            sentiment = "bullish"
        elif net_shares < -50000:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        result = {
            "period_days": 90,
            "total_transactions": len(filtered),
            "net_shares_bought": net_shares,
            "insider_sentiment": sentiment,
            "transactions": filtered,
        }
        self.logger.info(f"[{self.ticker}] Insider transactions: {len(filtered)} in last 90 days, sentiment={sentiment}")
        return result

    def fetch_all(self) -> dict:
        profile = {}
        financials = {}
        recommendations = {}
        earnings = {}
        insider = {}
        errors = []

        try:
            profile = self.fetch_company_profile()
        except Exception as exc:
            errors.append(f"company_profile: {exc}")
            self.logger.error(f"[{self.ticker}] company_profile failed: {exc}")

        try:
            financials = self.fetch_basic_financials()
        except Exception as exc:
            errors.append(f"basic_financials: {exc}")
            self.logger.error(f"[{self.ticker}] basic_financials failed: {exc}")

        try:
            recommendations = self.fetch_analyst_recommendations()
        except Exception as exc:
            errors.append(f"analyst_recommendations: {exc}")
            self.logger.error(f"[{self.ticker}] analyst_recommendations failed: {exc}")

        try:
            earnings = self.fetch_earnings_history()
        except Exception as exc:
            errors.append(f"earnings_history: {exc}")
            self.logger.error(f"[{self.ticker}] earnings_history failed: {exc}")

        try:
            insider = self.fetch_insider_transactions()
        except Exception as exc:
            errors.append(f"insider_transactions: {exc}")
            self.logger.error(f"[{self.ticker}] insider_transactions failed: {exc}")

        company_profile_out = {**profile, "errors": errors}
        key_metrics_out = {
            "ticker": self.ticker,
            "basic_financials": financials,
            "analyst_recommendations": recommendations,
            "earnings_history": earnings,
            "insider_transactions": insider,
            "errors": errors,
        }

        save_json(company_profile_out, f"{self.config.DATA_DIR}/{self.ticker}/company_profile.json")
        save_json(key_metrics_out, f"{self.config.DATA_DIR}/{self.ticker}/key_metrics.json")

        return {**company_profile_out, **key_metrics_out}
