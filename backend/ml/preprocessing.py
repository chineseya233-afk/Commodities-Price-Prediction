"""
Data Preprocessing Pipeline (Data Engineer Agent)

Handles all data cleaning, transformation, and quality checks
before feeding into the prediction models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for commodity price data.
    
    Processing steps:
    1. Missing value imputation (forward fill + linear interpolation)
    2. Outlier detection and handling (IQR + Z-Score)
    3. Data normalization
    4. Data quality reporting
    """

    def __init__(self, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.quality_report: dict = {}

    def process(self, df: pd.DataFrame, target_col: str = "price") -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        logger.info(f"Starting preprocessing pipeline. Shape: {df.shape}")
        
        # 第 1 步：按日期排序
        df = df.sort_values("date").reset_index(drop=True)
        
        # 第 2 步：处理缺失值
        df = self._handle_missing_values(df, target_col)
        
        # 第 3 步：检测并标记异常值
        df = self._detect_outliers(df, target_col)
        
        # 第 4 步：生成质量报告
        self._generate_quality_report(df, target_col)
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Forward fill then linear interpolation for remaining gaps."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 统计处理前缺失值
        missing_before = df[numeric_cols].isnull().sum().sum()
        
        # 先向前填充（使用最近已知值）
        df[numeric_cols] = df[numeric_cols].ffill()
        
        # 对剩余缺口做线性插值（序列起始处）
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        
        # 对剩余边界情况做向后填充
        df[numeric_cols] = df[numeric_cols].bfill()
        
        missing_after = df[numeric_cols].isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df

    def _detect_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Flag outliers using IQR and Z-Score methods."""
        if target_col not in df.columns:
            return df
            
        values = df[target_col].values
        
        # IQR 方法
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        iqr_outliers = (values < lower_bound) | (values > upper_bound)
        
        # Z-Score 方法
        mean = np.mean(values)
        std = np.std(values)
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            zscore_outliers = z_scores > self.zscore_threshold
        else:
            zscore_outliers = np.zeros(len(values), dtype=bool)
        
        # 合并结果：任一方法检测到即标记
        df["is_outlier"] = iqr_outliers | zscore_outliers
        
        outlier_count = df["is_outlier"].sum()
        logger.info(f"Outliers detected: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
        
        return df

    def _generate_quality_report(self, df: pd.DataFrame, target_col: str) -> None:
        """Generate a data quality summary report."""
        self.quality_report = {
            "total_records": len(df),
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
            "completeness": round((1 - df.isnull().sum().sum() / df.size) * 100, 2),
            "target_stats": {
                "mean": round(df[target_col].mean(), 2),
                "std": round(df[target_col].std(), 2),
                "min": round(df[target_col].min(), 2),
                "max": round(df[target_col].max(), 2),
                "median": round(df[target_col].median(), 2),
            },
            "outlier_count": int(df.get("is_outlier", pd.Series(dtype=bool)).sum()),
            "outlier_pct": round(
                df.get("is_outlier", pd.Series(dtype=bool)).sum() / len(df) * 100, 2
            ),
        }

    def get_quality_report(self) -> dict:
        """Return the data quality report."""
        return self.quality_report
