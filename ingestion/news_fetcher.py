import difflib
from datetime import datetime, timedelta, timezone

import finnhub
from newsapi import NewsApiClient

from config import Config
from ingestion.utils import rate_limit, retry_with_backoff, save_json, setup_logger

FINANCE_KEYWORDS = {
    "earnings", "revenue", "profit", "guidance", "forecast",
    "dividend", "acquisition", "merger", "lawsuit", "regulation",
    "fda", "sec", "buyback", "layoffs", "outlook",
}


class NewsFetcher:
    def __init__(self, ticker: str, company_name: str, config: Config):
        self.ticker = ticker.upper()
        self.company_name = company_name or ticker
        self.config = config
        self.logger = setup_logger("news_fetcher")
        self.finnhub_client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=config.NEWSAPI_KEY) if config.NEWSAPI_KEY else None

    def fetch_finnhub_news(self) -> list[dict]:
        self.logger.info(f"[{self.ticker}] Fetching news from Finnhub")
        rate_limit(self.config.FINNHUB_DELAY_SECONDS)

        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=self.config.NEWS_LOOKBACK_DAYS)

        def _fetch():
            return self.finnhub_client.company_news(
                self.ticker,
                _from=str(from_date),
                to=str(to_date),
            )

        raw = retry_with_backoff(_fetch) or []

        articles = []
        for item in raw:
            sentiment_score = item.get("sentiment", {}).get("bullishPercent") if isinstance(item.get("sentiment"), dict) else None
            if sentiment_score is None:
                sentiment_score = None
                sentiment_label = None
            else:
                sentiment_label = "positive" if sentiment_score >= 0.5 else "negative"

            ts = item.get("datetime")
            published = None
            if ts:
                try:
                    published = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    published = str(ts)

            articles.append({
                "source": "Finnhub",
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "url": item.get("url"),
                "published_at": published,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
            })

        self.logger.info(f"[{self.ticker}] Finnhub returned {len(articles)} articles")
        return articles

    def fetch_newsapi_articles(self) -> list[dict]:
        if not self.newsapi_client:
            self.logger.warning(f"[{self.ticker}] NewsAPI key not set, skipping")
            return []

        self.logger.info(f"[{self.ticker}] Fetching news from NewsAPI")
        rate_limit(0.5)

        from_dt = (datetime.now(timezone.utc) - timedelta(days=self.config.NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        query = f'"{self.company_name}" stock OR "{self.ticker}" earnings'

        def _fetch():
            return self.newsapi_client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                from_param=from_dt,
                page_size=20,
            )

        try:
            response = retry_with_backoff(_fetch)
        except Exception as exc:
            self.logger.error(f"[{self.ticker}] NewsAPI failed: {exc}")
            return []

        raw = (response or {}).get("articles", [])
        articles = []
        for item in raw:
            articles.append({
                "source": (item.get("source") or {}).get("name", "NewsAPI"),
                "headline": item.get("title"),
                "summary": item.get("description"),
                "url": item.get("url"),
                "published_at": item.get("publishedAt"),
                "sentiment_score": None,
                "sentiment_label": None,
            })

        self.logger.info(f"[{self.ticker}] NewsAPI returned {len(articles)} articles")
        return articles

    def deduplicate(self, articles: list[dict]) -> list[dict]:
        seen_urls: set[str] = set()
        seen_headlines: list[str] = []
        unique = []

        for article in articles:
            url = article.get("url") or ""
            headline = (article.get("headline") or "").lower()

            if url and url in seen_urls:
                continue

            duplicate = False
            for existing in seen_headlines:
                ratio = difflib.SequenceMatcher(None, headline, existing).ratio()
                if ratio > 0.85:
                    duplicate = True
                    break

            if not duplicate:
                if url:
                    seen_urls.add(url)
                seen_headlines.append(headline)
                unique.append(article)

        def sort_key(a: dict) -> str:
            return a.get("published_at") or ""

        unique.sort(key=sort_key, reverse=True)
        return unique

    def filter_relevant(self, articles: list[dict]) -> list[dict]:
        ticker_lower = self.ticker.lower()
        company_lower = self.company_name.lower()
        filtered = []

        for article in articles:
            headline = (article.get("headline") or "").strip()
            if not headline:
                continue

            text = (headline + " " + (article.get("summary") or "")).lower()

            if ticker_lower in text or company_lower in text:
                filtered.append(article)
                continue

            if any(kw in text for kw in FINANCE_KEYWORDS):
                filtered.append(article)

        return filtered

    def fetch_all(self) -> dict:
        finnhub_articles = []
        newsapi_articles = []

        try:
            finnhub_articles = self.fetch_finnhub_news()
        except Exception as exc:
            self.logger.error(f"[{self.ticker}] Finnhub news failed: {exc}")

        try:
            newsapi_articles = self.fetch_newsapi_articles()
        except Exception as exc:
            self.logger.error(f"[{self.ticker}] NewsAPI articles failed: {exc}")

        merged = finnhub_articles + newsapi_articles
        merged = self.deduplicate(merged)
        merged = self.filter_relevant(merged)

        output = {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lookback_days": self.config.NEWS_LOOKBACK_DAYS,
            "total_articles": len(merged),
            "articles": merged,
        }

        save_json(output, f"{self.config.DATA_DIR}/{self.ticker}/news_articles.json")
        self.logger.info(f"[{self.ticker}] Saved {len(merged)} news articles")
        return output
