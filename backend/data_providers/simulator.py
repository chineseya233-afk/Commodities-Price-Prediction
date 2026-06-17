"""
China Diesel Price Simulator (Data Engineer Agent)

Generates high-fidelity simulated 0# diesel prices for China domestic market.
The simulation is based on:
1. International crude oil price movements (Brent/WTI)
2. NDRC pricing mechanism (adjustments every 10 working days)
3. RMB/USD exchange rate effects
4. Tax structure (consumption tax + VAT)
5. Seasonal demand patterns
6. Random market noise (mean-reverting)

This provides realistic China-specific data for POC demonstration.
When real data sources become available, simply swap the DataProvider.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional

from .base import DataProvider


class ChinaDieselSimulator(DataProvider):
    """
    High-fidelity simulator for China 0# diesel prices.
    
    Pricing model:
    - Base price derived from international crude (Brent) with ~0.85 correlation
    - NDRC adjustment mechanism: price changes occur at ~10 working day intervals
    - Tax structure: consumption tax (1,411 RMB/ton) + 13% VAT
    - Seasonal pattern: higher in winter (heating) and summer (agriculture)
    - Regional spread: Fujian province premium/discount
    """

    # Fixed tax parameters (2024-2026)
    CONSUMPTION_TAX_PER_TON = 1411.0  # RMB/ton for diesel
    VAT_RATE = 0.13
    BARREL_TO_TON = 7.35  # approximate barrels per metric ton for diesel
    
    # Seasonal demand multipliers (monthly, 1-12)
    SEASONAL_FACTORS = {
        1: 1.03, 2: 1.01, 3: 0.98, 4: 0.97, 5: 0.98, 6: 1.01,
        7: 1.02, 8: 1.01, 9: 1.00, 10: 1.01, 11: 1.03, 12: 1.04
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._cached_data: Optional[pd.DataFrame] = None

    def get_provider_name(self) -> str:
        return "ChinaDieselSimulator"

    def get_supported_commodities(self) -> list[str]:
        return ["diesel_0"]

    async def get_latest_price(self, commodity: str) -> dict:
        """Get the latest simulated price."""
        df = await self.fetch_price_data(
            commodity, date.today() - timedelta(days=7), date.today()
        )
        if df.empty:
            return {"date": str(date.today()), "price": 7800.0, "change": 0, "change_pct": 0}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        change = latest["price"] - prev["price"]
        change_pct = (change / prev["price"]) * 100 if prev["price"] > 0 else 0
        return {
            "date": str(latest["date"]),
            "price": round(latest["price"], 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 4),
        }

    async def fetch_macro_indicators(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Generate simulated macro indicators."""
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        
        # Simulate Brent crude oil price (USD/barrel)
        brent_base = 78.0
        brent_returns = self.rng.normal(0, 0.015, n)
        brent_prices = brent_base * np.exp(np.cumsum(brent_returns))
        
        # Simulate USD/CNY exchange rate
        fx_base = 7.25
        fx_noise = self.rng.normal(0, 0.001, n)
        fx_rates = fx_base + np.cumsum(fx_noise)
        fx_rates = np.clip(fx_rates, 7.0, 7.6)
        
        # Simulate refinery utilization rate
        util_base = 0.78
        util_noise = self.rng.normal(0, 0.005, n)
        util_rates = util_base + np.cumsum(util_noise) * 0.1
        util_rates = np.clip(util_rates, 0.65, 0.92)
        
        records = []
        for i, d in enumerate(dates):
            records.extend([
                {"date": d.date(), "indicator_name": "brent_crude_usd", 
                 "value": round(brent_prices[i], 2), "source": "simulator"},
                {"date": d.date(), "indicator_name": "usd_cny_rate",
                 "value": round(fx_rates[i], 4), "source": "simulator"},
                {"date": d.date(), "indicator_name": "refinery_utilization",
                 "value": round(util_rates[i], 4), "source": "simulator"},
            ])
        
        return pd.DataFrame(records)

    async def fetch_price_data(
        self, commodity: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Generate simulated China 0# diesel prices.
        
        The simulation chain:
        1. Generate Brent crude oil path (GBM with mean reversion)
        2. Apply NDRC pricing formula: crude * conversion_factor * fx_rate + taxes
        3. Apply NDRC adjustment mechanism (step changes every ~10 working days)
        4. Add seasonal demand effects
        5. Add regional noise (Fujian province)
        """
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        
        if n == 0:
            return pd.DataFrame(columns=["date", "price", "open", "high", "low", 
                                          "volume", "commodity", "source"])

        # Step 1: Generate Brent crude path (mean-reverting GBM)
        brent_mean = 78.0  # long-term mean
        brent_vol = 0.018  # daily volatility
        mean_reversion_speed = 0.02
        
        brent = np.zeros(n)
        brent[0] = brent_mean + self.rng.normal(0, 3)
        for i in range(1, n):
            drift = mean_reversion_speed * (brent_mean - brent[i-1])
            shock = brent_vol * brent[i-1] * self.rng.normal()
            brent[i] = brent[i-1] + drift + shock
            brent[i] = max(brent[i], 50.0)  # floor

        # Step 2: Convert to RMB/ton base price
        fx_rate = 7.25 + np.cumsum(self.rng.normal(0, 0.002, n)) * 0.1
        fx_rate = np.clip(fx_rate, 7.0, 7.5)
        
        # Brent ($/barrel) -> RMB/ton
        base_price_rmb = brent * self.BARREL_TO_TON * fx_rate
        
        # Add taxes
        with_consumption_tax = base_price_rmb + self.CONSUMPTION_TAX_PER_TON
        with_vat = with_consumption_tax * (1 + self.VAT_RATE)

        # Step 3: Apply NDRC step adjustment mechanism
        # Price only changes every ~10 working days through NDRC adjustments
        ndrc_price = np.zeros(n)
        ndrc_price[0] = with_vat[0]
        adjustment_interval = 10
        last_adjustment_idx = 0
        
        for i in range(1, n):
            if (i - last_adjustment_idx) >= adjustment_interval:
                # NDRC adjustment window: compare current market price to current NDRC price
                market_change_pct = (with_vat[i] - ndrc_price[i-1]) / ndrc_price[i-1]
                
                # NDRC usually adjusts by 50-200 RMB/ton per step
                if abs(market_change_pct) > 0.005:  # >0.5% threshold triggers adjustment
                    adjustment = np.clip(
                        with_vat[i] - ndrc_price[i-1],
                        -200, 200
                    )
                    ndrc_price[i] = ndrc_price[i-1] + adjustment
                    last_adjustment_idx = i
                else:
                    ndrc_price[i] = ndrc_price[i-1]
            else:
                ndrc_price[i] = ndrc_price[i-1]

        # Step 4: Apply seasonal factors
        seasonal_mult = np.array([
            self.SEASONAL_FACTORS[d.month] for d in dates
        ])
        seasonal_price = ndrc_price * seasonal_mult

        # Step 5: Add Fujian regional noise
        regional_noise = self.rng.normal(0, 15, n)  # ±15 RMB/ton regional spread
        final_price = seasonal_price + regional_noise

        # Generate OHLV data (intra-day simulation)
        daily_range = np.abs(self.rng.normal(0, 25, n))  # typical daily range ~25 RMB
        high = final_price + daily_range * 0.6
        low = final_price - daily_range * 0.4
        open_price = low + (high - low) * self.rng.uniform(0.3, 0.7, n)
        volume = self.rng.uniform(8000, 15000, n).astype(int)

        df = pd.DataFrame({
            "date": [d.date() for d in dates],
            "price": np.round(final_price, 2),
            "open": np.round(open_price, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "volume": volume,
            "commodity": "diesel_0",
            "source": "simulator",
        })

        return df


class MockERPProvider(DataProvider):
    """
    Mock provider for internal ERP data simulation.
    
    Generates realistic procurement orders, inventory levels,
    and contract data to ensure the system architecture is complete.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def get_provider_name(self) -> str:
        return "MockERP"

    def get_supported_commodities(self) -> list[str]:
        return ["diesel_0"]

    async def get_latest_price(self, commodity: str) -> dict:
        return {
            "date": str(date.today()),
            "price": 7850.0,
            "change": -25.0,
            "change_pct": -0.32,
        }

    async def fetch_price_data(
        self, commodity: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Generate mock procurement price records."""
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        
        base = 7800 + np.cumsum(self.rng.normal(0, 20, n))
        
        return pd.DataFrame({
            "date": [d.date() for d in dates],
            "price": np.round(base, 2),
            "open": np.round(base - self.rng.uniform(10, 30, n), 2),
            "high": np.round(base + self.rng.uniform(10, 50, n), 2),
            "low": np.round(base - self.rng.uniform(10, 50, n), 2),
            "volume": self.rng.randint(5000, 12000, n),
            "commodity": "diesel_0",
            "source": "mock_erp",
        })

    async def fetch_macro_indicators(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Return empty — ERP doesn't provide macro indicators."""
        return pd.DataFrame(columns=["date", "indicator_name", "value", "source"])
