"""Date utilities shared by API and tests."""

from datetime import date, timedelta
from typing import Any, Iterable, List, Mapping, Tuple


def training_start_date(end_date: date) -> date:
    """Return a training start date that includes the 2024-2025 validation window."""
    if end_date >= date(2026, 1, 1):
        return date(2024, 1, 1)
    return end_date - timedelta(days=730)


def build_calendar_dates(start: date, horizon: int) -> List[date]:
    """Build forecast dates as calendar days, including the start date."""
    return [start + timedelta(days=offset) for offset in range(max(int(horizon), 0))]


def build_visible_forecast_targets(
    data_end_date: date,
    display_start_date: date,
    horizon: int,
) -> List[Tuple[int, date]]:
    """Map forecast array offsets to visible calendar target dates.

    Model output index 0 forecasts the first calendar day after the latest real
    data point. If the market data feed lags behind today, older forecast
    offsets are hidden instead of being relabeled to today.
    """
    forecast_dates = build_calendar_dates(data_end_date + timedelta(days=1), horizon)
    visible = [
        (index, target_date)
        for index, target_date in enumerate(forecast_dates)
        if target_date >= display_start_date
    ]
    if visible:
        return visible
    # 如果真实市场数据源滞后时间超过预测窗口，仍然展示
    # 锚定到最新可用数据的预测窗口，而不是返回
    # 空预测并迫使前端显示零值摘要。
    return list(enumerate(forecast_dates))


def build_business_dates(start: date, horizon: int) -> List[date]:
    """Build the next unique business dates after start."""
    dates: List[date] = []
    cursor = start
    while len(dates) < horizon:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            dates.append(cursor)
    return dates


def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def build_calendar_price_history(
    rows: Iterable[Mapping[str, Any]],
    end_date: date,
    max_days: int = 90,
) -> List[dict]:
    """Return calendar-day history, forward-filling non-trading day prices."""
    sorted_rows = sorted((dict(row) for row in rows), key=lambda item: _to_date(item["date"]))
    if not sorted_rows:
        return []

    by_date = {_to_date(row["date"]): row for row in sorted_rows}
    first_date = max(_to_date(sorted_rows[0]["date"]), end_date - timedelta(days=max(int(max_days), 1) - 1))
    last_known = None
    history: List[dict] = []
    cursor = first_date

    while cursor <= end_date:
        row = by_date.get(cursor)
        if row is not None:
            last_known = row
        if last_known is not None:
            history.append({
                "date": str(cursor),
                "price": float(round(float(last_known["price"]), 2)),
                "high": float(round(float(last_known.get("high", last_known["price"])), 2)),
                "low": float(round(float(last_known.get("low", last_known["price"])), 2)),
            })
        cursor += timedelta(days=1)

    return history[-max(int(max_days), 1):]
