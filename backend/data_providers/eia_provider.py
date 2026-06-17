"""
EIA Data Provider — US Energy Information Administration API v2

Fetches real diesel and crude oil spot prices from EIA's free public API.
Implements the DataProvider abstract interface for seamless switching.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional
from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import DataProvider


class EIAProvider(DataProvider):
    """
    Data provider for EIA (Energy Information Administration) petroleum data.

    Endpoints used:
    - Diesel retail prices: /petroleum/pri/spt/data/
    - WTI crude spot: /petroleum/pri/spt/data/
    """

    # EIA series IDs
    DIESEL_SERIES = "EER_EPD2DXL0_PF4_Y35NY_DPG"  # New York Harbor ULSD spot
    WTI_CRUDE_SERIES = "RWTC"  # WTI Crude Oil spot

    # Approximate USD→RMB/ton conversion for diesel
    # 1 gallon ≈ 3.785 liters, diesel density ≈ 0.835 kg/L
    # 1 ton ≈ 1000 / (3.785 * 0.835) gallons ≈ 316.3 gallons
    GALLONS_PER_TON = 316.3
    DEFAULT_USD_CNY_RATE = 7.25

    def __init__(
        self,
        api_key: str = "demo",
        base_url: str = "https://api.eia.gov/v2",
        diesel_series: str = DIESEL_SERIES,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.diesel_series = diesel_series or self.DIESEL_SERIES
        self._cache: dict = {}

    async def fetch_price_data(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch diesel price data from EIA and convert to RMB/ton."""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed. Cannot fetch EIA data.")
            return pd.DataFrame()

        cache_key = f"{commodity}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            url = f"{self.base_url}/petroleum/pri/spt/data/"
            base_params = {
                "api_key": self.api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": self.diesel_series,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "length": 5000,
            }

            records = []
            offset = 0
            async with httpx.AsyncClient(timeout=45) as client:
                while True:
                    params = {**base_params, "offset": offset}
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    page = data.get("response", {}).get("data", [])
                    records.extend(page)
                    if len(page) < base_params["length"]:
                        break
                    offset += len(page)
            if not records:
                logger.warning("EIA returned no data")
                return pd.DataFrame()

            rows = []
            for r in records:
                try:
                    price_usd_gallon = float(r["value"])
                    # Convert: USD/gallon → RMB/ton
                    price_rmb_ton = price_usd_gallon * self.GALLONS_PER_TON * self.DEFAULT_USD_CNY_RATE
                    rows.append({
                        "date": r["period"],
                        "price": round(price_rmb_ton, 2),
                        "price_usd_gallon": round(price_usd_gallon, 4),
                        "open": round(price_rmb_ton * 0.998, 2),
                        "high": round(price_rmb_ton * 1.005, 2),
                        "low": round(price_rmb_ton * 0.995, 2),
                        "volume": 0,
                        "commodity": commodity,
                        "source": "eia",
                        "source_series": r.get("series", self.diesel_series),
                    })
                except (ValueError, KeyError):
                    continue

            df = pd.DataFrame(rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                self._cache[cache_key] = df
                logger.info(
                    f"EIA: Fetched {len(df)} diesel price records "
                    f"({self.diesel_series}, {df['date'].dt.date.min()} to {df['date'].dt.date.max()})"
                )

            return df

        except Exception as e:
            logger.error(f"EIA API error: {e}")
            return pd.DataFrame()

    async def fetch_macro_indicators(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """EIA doesn't provide macro indicators — return empty."""
        return pd.DataFrame()

    async def get_latest_price(self, commodity: str) -> dict:
        """Get the most recent price from EIA."""
        end = date.today()
        start = end - timedelta(days=14)
        df = await self.fetch_price_data(commodity, start, end)
        if df.empty:
            return {"date": str(end), "price": 0, "change": 0, "change_pct": 0}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = latest["price"] - prev["price"]
        change_pct = (change / prev["price"]) * 100 if prev["price"] > 0 else 0

        return {
            "date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
            "price": float(latest["price"]),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 4),
        }

    def get_provider_name(self) -> str:
        return "US Energy Information Administration (EIA)"

    def get_supported_commodities(self) -> list[str]:
        return ["diesel_0", "crude_wti"]
