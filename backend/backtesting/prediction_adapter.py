"""
Forecast-to-signal adapter for procurement backtests.

The adapter only consumes forecast-time inputs. It must not inspect realized
future prices when creating a signal.
"""

import math
from datetime import date, datetime, time
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from .schemas import ProcurementAction, Signal


class ForecastSignalConfig(BaseModel):
    """Deterministic thresholds for mapping forecasts to procurement actions."""

    buy_threshold_pct: float = Field(default=0.02, ge=0.0)
    lock_threshold_pct: float = Field(default=0.05, ge=0.0)
    defer_threshold_pct: float = Field(default=-0.02, le=0.0)
    wide_interval_pct: float = Field(default=0.12, ge=0.0)
    model_disagreement_pct: float = Field(default=0.08, ge=0.0)
    default_quantity: float = Field(default=1.0, gt=0.0)
    no_forecast_action: ProcurementAction = ProcurementAction.HOLD
    qa_fail_action: ProcurementAction = ProcurementAction.REQUEST_QUOTE
    wide_interval_action: ProcurementAction = ProcurementAction.REQUEST_QUOTE
    disagreement_action: ProcurementAction = ProcurementAction.HEDGE_REVIEW


_QA_FAIL_STATUSES = {
    "blocked",
    "fail",
    "failed",
    "false",
    "invalid",
    "red",
    "reject",
    "rejected",
}


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError("decision_time must be a datetime, date, or ISO datetime string")


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_prices(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        number = _finite_float(values)
        return [number] if number is not None else []

    prices: List[float] = []
    for value in values:
        number = _finite_float(value)
        if number is not None:
            prices.append(number)
    return prices


def _qa_failed(qa_passed: Optional[bool], qa_status: Optional[str]) -> bool:
    if qa_passed is False:
        return True
    if qa_status is None:
        return False
    return str(qa_status).strip().lower() in _QA_FAIL_STATUSES


def _resolve_interval_width_pct(
    current_price: Optional[float],
    low_price: Optional[float],
    high_price: Optional[float],
) -> Optional[float]:
    if current_price is None or current_price <= 0:
        return None
    if low_price is None or high_price is None:
        return None
    lower, upper = sorted((low_price, high_price))
    return max(0.0, (upper - lower) / current_price)


def _resolve_disagreement_pct(
    current_price: Optional[float],
    model_disagreement: Optional[float],
    model_disagreement_pct: Optional[float],
) -> Optional[float]:
    explicit_pct = _finite_float(model_disagreement_pct)
    if explicit_pct is not None:
        return abs(explicit_pct)

    raw_value = _finite_float(model_disagreement)
    if raw_value is None:
        return None
    raw_value = abs(raw_value)
    if raw_value <= 1.0:
        return raw_value
    if current_price is not None and current_price > 0:
        return raw_value / current_price
    return raw_value


def _confidence(
    change_pct: Optional[float],
    interval_width_pct: Optional[float],
    config: ForecastSignalConfig,
) -> Optional[float]:
    if change_pct is None:
        return None
    denominator = max(config.buy_threshold_pct, abs(config.defer_threshold_pct), 1e-12)
    score = min(abs(change_pct) / denominator, 1.0)
    if interval_width_pct is not None and config.wide_interval_pct > 0:
        penalty = min(interval_width_pct / config.wide_interval_pct, 1.0) * 0.5
        score *= 1.0 - penalty
    return max(0.0, min(1.0, score))


def forecast_to_signal(
    *,
    current_price: Any,
    decision_time: Any,
    forecast_horizon_days: int = 1,
    instrument: str = "commodity",
    predicted_prices: Optional[Iterable[Any]] = None,
    p50: Any = None,
    p10: Any = None,
    p90: Any = None,
    predicted_p50: Any = None,
    predicted_p10: Any = None,
    predicted_p90: Any = None,
    qa_passed: Optional[bool] = True,
    qa_status: Optional[str] = None,
    model_disagreement: Any = None,
    model_disagreement_pct: Any = None,
    quantity: Any = None,
    config: Optional[ForecastSignalConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Signal:
    """Convert forecast inputs into a procurement Signal.

    Directional rules are deterministic:
    - clearly higher forecast: buy now or lock contract
    - clearly lower forecast: defer
    - wide interval, failed QA, or model disagreement: risk-review action
    - otherwise: hold
    """

    signal_config = config or ForecastSignalConfig()
    decision_dt = _coerce_datetime(decision_time)
    current = _finite_float(current_price)
    forecast_points = _finite_prices(predicted_prices)

    median_price = _finite_float(p50)
    if median_price is None:
        median_price = _finite_float(predicted_p50)
    if median_price is None and forecast_points:
        median_price = forecast_points[-1]

    low_price = _finite_float(p10)
    if low_price is None:
        low_price = _finite_float(predicted_p10)
    high_price = _finite_float(p90)
    if high_price is None:
        high_price = _finite_float(predicted_p90)

    interval_width_pct = _resolve_interval_width_pct(current, low_price, high_price)
    disagreement_pct = _resolve_disagreement_pct(
        current, model_disagreement, model_disagreement_pct
    )

    change_pct: Optional[float] = None
    if current is not None and current > 0 and median_price is not None:
        change_pct = (median_price - current) / current

    action = ProcurementAction.HOLD
    reason = "insignificant_change"

    if current is None or current <= 0:
        action = signal_config.no_forecast_action
        reason = "invalid_current_price"
    elif _qa_failed(qa_passed, qa_status):
        action = signal_config.qa_fail_action
        reason = "qa_failed"
    elif median_price is None:
        action = signal_config.no_forecast_action
        reason = "missing_forecast"
    elif (
        interval_width_pct is not None
        and interval_width_pct > signal_config.wide_interval_pct
    ):
        action = signal_config.wide_interval_action
        reason = "wide_forecast_interval"
    elif (
        disagreement_pct is not None
        and disagreement_pct > signal_config.model_disagreement_pct
    ):
        action = signal_config.disagreement_action
        reason = "model_disagreement"
    elif change_pct is not None and change_pct >= max(
        signal_config.lock_threshold_pct, signal_config.buy_threshold_pct
    ):
        action = ProcurementAction.LOCK_CONTRACT
        reason = "forecast_strongly_up"
    elif change_pct is not None and change_pct >= signal_config.buy_threshold_pct:
        action = ProcurementAction.BUY_NOW
        reason = "forecast_up"
    elif change_pct is not None and change_pct <= signal_config.defer_threshold_pct:
        action = ProcurementAction.DEFER
        reason = "forecast_down"

    signal_quantity = _finite_float(quantity)
    if signal_quantity is None or signal_quantity <= 0:
        signal_quantity = signal_config.default_quantity

    signal_metadata: Dict[str, Any] = dict(metadata or {})
    signal_metadata.update(
        {
            "reason": reason,
            "p50": median_price,
            "p10": low_price,
            "p90": high_price,
            "expected_change_pct": change_pct,
            "interval_width_pct": interval_width_pct,
            "model_disagreement_pct": disagreement_pct,
            "qa_status": qa_status,
            "qa_passed": qa_passed,
            "prediction_point_count": len(forecast_points),
            "uses_realized_prices": False,
        }
    )

    return Signal(
        instrument=instrument,
        decision_time=decision_dt,
        action=action,
        forecast_horizon_days=max(0, int(forecast_horizon_days or 0)),
        signal_id=f"{instrument}:{decision_dt.isoformat()}:{forecast_horizon_days}",
        expected_price=median_price,
        reference_price=current,
        quantity=signal_quantity,
        confidence=_confidence(change_pct, interval_width_pct, signal_config),
        metadata=signal_metadata,
    )
