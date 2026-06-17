"""
Feature Engineering Module (Data Engineer Agent)

Creates all features required by the prediction models:
- Calendar features (day of week, month, holidays, NDRC adjustment windows)
- Technical indicators (MA, RSI, Bollinger Bands)
- Lag features and rolling statistics
- External covariate alignment
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger


class FeatureEngineer:
    """
    Comprehensive feature engineering for commodity price prediction.
    
    Produces features aligned with TFT's variable classification:
    - Known future: calendar, scheduled events
    - Unknown future: price-derived technicals, external covariates
    - Static: commodity type, region
    """

    # 中国公众假期（POC 简化版）
    CHINESE_HOLIDAYS = {
        (1, 1), (1, 2), (1, 3),           # New Year
        (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17),  # Spring Festival (approximate)
        (4, 4), (4, 5), (4, 6),            # Qingming
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),  # Labor Day
        (6, 8), (6, 9), (6, 10),           # Dragon Boat
        (9, 15), (9, 16), (9, 17),         # Mid-Autumn
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),  # National Day
    }

    def __init__(self, ma_windows: List[int] = None, rsi_window: int = 14):
        self.ma_windows = ma_windows or [5, 10, 20]
        self.rsi_window = rsi_window

    def create_features(self, df: pd.DataFrame, target_col: str = "price") -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # 日历特征（未来已知）
        df = self._add_calendar_features(df)
        
        # 技术指标（未来未知）
        df = self._add_technical_indicators(df, target_col)
        
        # 滞后特征（未来未知）
        df = self._add_lag_features(df, target_col)
        
        # 滚动统计（未来未知）
        df = self._add_rolling_stats(df, target_col)
        
        # 价格变化特征
        df = self._add_price_change_features(df, target_col)
        
        # 发改委调价窗口指标
        df = self._add_ndrc_window(df)
        
        # 填充特征列中的 NaN，而不是直接删除行
        initial_len = len(df)
        feature_cols = [c for c in df.columns if c not in ["date", "commodity", "source"]]
        df[feature_cols] = df[feature_cols].ffill().bfill()
        # 只删除目标值本身仍为 NaN 的行
        remaining_na = df.isnull().any(axis=1).sum()
        if remaining_na > 0:
            df = df.dropna().reset_index(drop=True)
        logger.info(f"Feature engineering complete. {initial_len} → {len(df)} rows ({initial_len - len(df)} dropped for NaN)")
        
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add known-future calendar features."""
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.values.astype(int)
        try:
            df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        except AttributeError:
            df["is_month_start"] = (df["date"].dt.day == 1).astype(int)
            df["is_month_end"] = ((df["date"] + pd.Timedelta(days=1)).dt.day == 1).astype(int)
        
        # 节假日指标
        df["is_holiday"] = df["date"].apply(
            lambda x: 1 if (x.month, x.day) in self.CHINESE_HOLIDAYS else 0
        )
        
        # 节假日前后日期（市场反应）
        df["is_near_holiday"] = (
            df["is_holiday"].shift(1).fillna(0).astype(int) | 
            df["is_holiday"].shift(-1).fillna(0).astype(int)
        )
        
        return df

    def _add_technical_indicators(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add technical analysis indicators."""
        prices = df[target_col]
        
        # 移动均线
        for window in self.ma_windows:
            df[f"ma_{window}"] = prices.rolling(window=window).mean()
            # 价格相对均线位置
            df[f"price_vs_ma_{window}"] = (prices - df[f"ma_{window}"]) / df[f"ma_{window}"]
        
        # RSI（相对强弱指数）
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # 布林带（20 日）
        bb_window = 20
        df["bb_middle"] = prices.rolling(bb_window).mean()
        bb_std = prices.rolling(bb_window).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (prices - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # 波动率（收益率 20 日滚动标准差）
        returns = prices.pct_change()
        df["volatility_20d"] = returns.rolling(20).std()
        
        return df

    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add lagged price values."""
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            df[f"price_lag_{lag}"] = df[target_col].shift(lag)
        return df

    def _add_rolling_stats(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add rolling window statistics."""
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = df[target_col].rolling(window).mean()
            df[f"rolling_std_{window}"] = df[target_col].rolling(window).std()
            df[f"rolling_min_{window}"] = df[target_col].rolling(window).min()
            df[f"rolling_max_{window}"] = df[target_col].rolling(window).max()
            df[f"rolling_range_{window}"] = df[f"rolling_max_{window}"] - df[f"rolling_min_{window}"]
        return df

    def _add_price_change_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add price change and momentum features."""
        df["price_change_1d"] = df[target_col].diff(1)
        df["price_change_pct_1d"] = df[target_col].pct_change(1) * 100
        df["price_change_5d"] = df[target_col].diff(5)
        df["price_change_pct_5d"] = df[target_col].pct_change(5) * 100
        df["price_change_20d"] = df[target_col].diff(20)
        df["price_change_pct_20d"] = df[target_col].pct_change(20) * 100
        
        # 动量：当前价格相对 N 天前
        df["momentum_5d"] = df[target_col] / df[target_col].shift(5)
        df["momentum_10d"] = df[target_col] / df[target_col].shift(10)
        
        return df

    def _add_ndrc_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add NDRC adjustment window indicator (10 working day cycle)."""
        # 从起点开始累计的工作日数
        df["working_day_idx"] = range(len(df))
        # 发改委 10 工作日周期内的位置
        df["ndrc_cycle_position"] = df["working_day_idx"] % 10
        # 接近调价窗口（周期最后 2 天）
        df["is_near_adjustment"] = (df["ndrc_cycle_position"] >= 8).astype(int)
        return df

    def get_known_future_columns(self) -> List[str]:
        """Return column names classified as known future variables for TFT."""
        return [
            "day_of_week", "month", "quarter", "day_of_month", "week_of_year",
            "is_month_start", "is_month_end", "is_holiday", "is_near_holiday",
            "ndrc_cycle_position", "is_near_adjustment",
        ]

    def get_unknown_future_columns(self) -> List[str]:
        """Return column names classified as unknown future variables for TFT."""
        cols = []
        for w in self.ma_windows:
            cols.extend([f"ma_{w}", f"price_vs_ma_{w}"])
        cols.extend([
            "rsi", "bb_width", "bb_position", "volatility_20d",
            "price_change_1d", "price_change_pct_1d",
            "price_change_5d", "price_change_pct_5d",
            "momentum_5d", "momentum_10d",
        ])
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            cols.append(f"price_lag_{lag}")
        for window in [7, 14, 30]:
            cols.extend([
                f"rolling_mean_{window}", f"rolling_std_{window}",
                f"rolling_range_{window}",
            ])
        return cols
