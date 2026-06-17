"""Date-based train/test split helpers for model validation."""

from datetime import date
from typing import Tuple

import pandas as pd


def select_train_test_by_date(
    df: pd.DataFrame,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by explicit calendar windows without leaking test data."""
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    train_mask = (
        (work["date"].dt.date >= train_start)
        & (work["date"].dt.date <= train_end)
    )
    test_mask = (
        (work["date"].dt.date >= test_start)
        & (work["date"].dt.date <= test_end)
    )
    return (
        work.loc[train_mask].sort_values("date").reset_index(drop=True),
        work.loc[test_mask].sort_values("date").reset_index(drop=True),
    )
