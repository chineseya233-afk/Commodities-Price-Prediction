"""
FRED Data Provider — Federal Reserve Economic Data

Fetches macroeconomic indicators relevant to commodity pricing:
- USD/CNY exchange rate
- Brent crude oil prices
- Federal Funds Rate
"""

import pandas as pd
from datetime import date, timedelta
from typing import Optional
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import DataProvider


class FREDProvider(DataProvider):
    """
    Data provider for FRED (Federal Reserve Economic Data).

    Provides macro indicators as additional features for the prediction models.
    """

    # FRED series IDs relevant to diesel pricing
    SERIES = {
        "DEXCHUS": "USD/CNY Exchange Rate",
        "DCOILBRENTEU": "Brent Crude Oil (USD/barrel)",
        "DFF": "Federal Funds Rate",
    }

    def __init__(self, api_key: str = "", base_url: str = "https://api.stlouisfed.org/fred"):
        self.api_key = api_key
        self.base_url = base_url
        self._cache: dict = {}

    async def _fetch_series(self, series_id: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch a single FRED series."""
        if not HTTPX_AVAILABLE:
            return pd.DataFrame()

        cache_key = f"{series_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            url = f"{self.base_url}/series/observations"
            params = {
                "series_id": series_id,
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end": end_date.strftime("%Y-%m-%d"),
                "file_type": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            observations = data.get("observations", [])
            rows = []
            for obs in observations:
                if obs.get("value", ".") != ".":
                    try:
                        rows.append({
                            "date": obs["date"],
                            "value": float(obs["value"]),
                        })
                    except (ValueError, KeyError):
                        continue

            df = pd.DataFrame(rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                self._cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return pd.DataFrame()

    async def fetch_price_data(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """FRED doesn't provide commodity prices directly — return empty."""
        return pd.DataFrame()

    async def fetch_macro_indicators(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch all macro indicators and return as a wide-format DataFrame."""
        result_df = None

        for series_id, description in self.SERIES.items():
            df = await self._fetch_series(series_id, start_date, end_date)
            if df.empty:
                continue

            df = df.rename(columns={"value": series_id.lower()})

            if result_df is None:
                result_df = df
            else:
                result_df = pd.merge(result_df, df, on="date", how="outer")

            logger.info(f"FRED: Fetched {len(df)} records for {description}")

        if result_df is not None:
            result_df = result_df.sort_values("date").reset_index(drop=True)
            # Forward-fill gaps (weekends, holidays)
            result_df = result_df.ffill().bfill()
            return result_df

        return pd.DataFrame()

    async def get_latest_price(self, commodity: str) -> dict:
        """Not applicable for FRED — return empty."""
        return {"date": str(date.today()), "price": 0, "change": 0, "change_pct": 0}

    def get_provider_name(self) -> str:
        return "Federal Reserve Economic Data (FRED)"

    def get_supported_commodities(self) -> list[str]:
        return []  # FRED provides macro indicators, not commodity prices
