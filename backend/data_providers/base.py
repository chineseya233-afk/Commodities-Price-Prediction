"""Abstract base class for all data providers.

All data sources (EIA, mock, simulator) implement this interface,
enabling seamless switching between data sources.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for commodity data providers.
    
    Every data source in the system must implement this interface.
    This ensures that the prediction engine and API layer are completely
    decoupled from the underlying data source, allowing one-click switching
    from mock/simulated data to paid premium data sources (e.g., 卓创/隆众)
    in production.
    """

    @abstractmethod
    async def fetch_price_data(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a commodity.

        Returns:
            DataFrame with columns: ['date', 'price', 'open', 'high', 'low', 
                                      'volume', 'commodity', 'source']
        """
        pass

    @abstractmethod
    async def fetch_macro_indicators(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators relevant to commodity pricing.

        Returns:
            DataFrame with columns: ['date', 'indicator_name', 'value', 'source']
        """
        pass

    @abstractmethod
    async def get_latest_price(self, commodity: str) -> dict:
        """
        Get the most recent price for a commodity.

        Returns:
            dict with keys: 'date', 'price', 'change', 'change_pct'
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this data provider."""
        pass

    @abstractmethod
    def get_supported_commodities(self) -> list[str]:
        """Return list of commodity codes supported by this provider."""
        pass
