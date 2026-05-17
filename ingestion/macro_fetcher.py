from datetime import datetime, timezone

import requests

from config import Config
from ingestion.utils import rate_limit, retry_with_backoff, save_json, setup_logger


class MacroFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("macro_fetcher")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.SEC_USER_AGENT})

    def fetch_series(self, series_id: str, series_name: str, limit: int = 24) -> dict:
        self.logger.info(f"Fetching FRED series: {series_id} ({series_name})")
        rate_limit(self.config.FRED_DELAY_SECONDS)

        def _fetch():
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.config.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            }
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()

        raw = retry_with_backoff(_fetch)
        observations_raw = raw.get("observations", [])

        observations = []
        for obs in observations_raw:
            val_str = obs.get("value", ".")
            try:
                val = float(val_str)
            except ValueError:
                continue
            observations.append({"date": obs.get("date"), "value": val})

        observations.reverse()

        latest_value = observations[-1]["value"] if observations else None
        latest_date = observations[-1]["date"] if observations else None

        units = ""
        try:
            units_url = "https://api.stlouisfed.org/fred/series"
            units_params = {
                "series_id": series_id,
                "api_key": self.config.FRED_API_KEY,
                "file_type": "json",
            }
            rate_limit(self.config.FRED_DELAY_SECONDS)
            units_resp = self.session.get(units_url, params=units_params, timeout=15)
            if units_resp.ok:
                series_data = units_resp.json().get("seriess", [{}])
                units = series_data[0].get("units_short", "") if series_data else ""
        except Exception:
            pass

        trend = self.calculate_trend(observations)

        return {
            "series_id": series_id,
            "name": series_name,
            "unit": units,
            "latest_value": latest_value,
            "latest_date": latest_date,
            "trend_3m": trend,
            "observations": observations,
        }

    def calculate_trend(self, observations: list[dict]) -> str:
        if len(observations) < 4:
            return "flat"
        latest = observations[-1]["value"]
        three_months_ago = observations[-4]["value"]
        if three_months_ago == 0:
            return "flat"
        change_pct = (latest - three_months_ago) / abs(three_months_ago) * 100
        if change_pct > 5:
            return "up"
        elif change_pct < -5:
            return "down"
        return "flat"

    def _build_macro_summary(self, indicators: dict) -> str:
        parts = []
        for name, data in indicators.items():
            val = data.get("latest_value")
            unit = data.get("unit", "")
            trend = data.get("trend_3m", "flat")
            if val is not None:
                label = name.replace("_", " ").title()
                parts.append(f"{label}: {val:.2f}{' ' + unit if unit else ''} (trend: {trend})")
        if not parts:
            return "Macro data unavailable."
        return "Current macroeconomic conditions — " + "; ".join(parts) + "."

    def fetch_all(self, ticker: str = "global") -> dict:
        indicators = {}

        for series_name, series_id in self.config.FRED_SERIES.items():
            try:
                indicators[series_name] = self.fetch_series(series_id, series_name, limit=24)
            except Exception as exc:
                self.logger.error(f"FRED series {series_id} ({series_name}) failed: {exc}")
                indicators[series_name] = {"series_id": series_id, "name": series_name, "error": str(exc)}

        macro_summary = self._build_macro_summary(indicators)

        output = {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "indicators": indicators,
            "macro_summary": macro_summary,
        }

        save_json(output, f"{self.config.DATA_DIR}/{ticker}/macro_indicators.json")
        self.logger.info(f"Macro indicators saved ({len(indicators)} series)")
        return output
