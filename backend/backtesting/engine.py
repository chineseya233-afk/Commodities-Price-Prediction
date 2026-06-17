"""
Procurement backtest engine built on the backtesting foundation contracts.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from .metrics import calculate_procurement_savings, calculate_strategy_metrics
from .prediction_adapter import (
    ForecastSignalConfig,
    _coerce_datetime,
    _finite_float,
    _finite_prices,
    forecast_to_signal,
)
from .schemas import (
    BacktestConfig,
    BacktestResult,
    EquityPoint,
    ExecutionTiming,
    Fill,
    Order,
    ProcurementAction,
    Signal,
)


class ProcurementBacktestResult(BacktestResult):
    """Backtest result with procurement-specific top-level details."""

    procurement_savings: Dict[str, float] = Field(default_factory=dict)
    period_results: List[Dict[str, Any]] = Field(default_factory=list)


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _first_present(item: Any, *names: str) -> Any:
    for name in names:
        value = _read_field(item, name)
        if value is not None:
            return value
    return None


def _safe_datetime(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    try:
        return _coerce_datetime(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


def _as_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _periods_from_input(periods: Any) -> List[Any]:
    if periods is None:
        return []
    if isinstance(periods, dict):
        nested_periods = periods.get("periods")
        if nested_periods is not None:
            return _as_sequence(nested_periods)
        if any(
            key in periods
            for key in (
                "current_price",
                "actual_prices",
                "predicted_prices",
                "decision_time",
            )
        ):
            return [periods]
        return [
            value
            for value in periods.values()
            if isinstance(value, dict) or hasattr(value, "current_price")
        ]
    return _as_sequence(periods)


def _coerce_backtest_config(config: Any) -> BacktestConfig:
    if config is None:
        return BacktestConfig()
    if isinstance(config, BacktestConfig):
        return config
    if isinstance(config, dict):
        return BacktestConfig(**config)
    return config


def _coerce_signal_config(config: Any) -> ForecastSignalConfig:
    if config is None:
        return ForecastSignalConfig()
    if isinstance(config, ForecastSignalConfig):
        return config
    if isinstance(config, dict):
        return ForecastSignalConfig(**config)
    return config


def _resolve_dates(period: Any) -> List[datetime]:
    dates: List[datetime] = []
    for value in _as_sequence(_read_field(period, "dates")):
        try:
            dates.append(_coerce_datetime(value))
        except (TypeError, ValueError):
            continue
    return dates


def _resolve_decision_time(
    period: Any, dates: List[datetime], period_index: int
) -> datetime:
    fallback = dates[0] if dates else datetime(1970, 1, 1) + timedelta(days=period_index)
    return _safe_datetime(_read_field(period, "decision_time"), fallback)


def _resolve_forecast_horizon(
    period: Any, actual_prices: List[float], dates: List[datetime]
) -> int:
    explicit = _read_field(period, "forecast_horizon_days")
    if explicit is not None:
        return _safe_int(explicit, 1)
    if len(dates) >= 2:
        return max(1, (dates[-1].date() - dates[0].date()).days)
    return max(1, len(actual_prices))


def _resolve_execution_time(
    period: Any,
    decision_time: datetime,
    dates: List[datetime],
    action: ProcurementAction,
    realized_price_time: datetime,
) -> Optional[datetime]:
    explicit = _read_field(period, "execution_time")
    if explicit is not None:
        explicit_time = _safe_datetime(explicit, decision_time)
        if explicit_time > decision_time:
            return explicit_time
    if action == ProcurementAction.DEFER:
        return realized_price_time if realized_price_time > decision_time else None

    future_dates = [value for value in dates if value > decision_time]
    if future_dates:
        return future_dates[0]
    return None


def _resolve_realized_price(
    period: Any,
    actual_prices: List[float],
    dates: List[datetime],
    decision_time: datetime,
    forecast_horizon_days: int,
) -> Tuple[Optional[float], datetime]:
    length_mismatch = len(dates) != len(actual_prices)
    fallback_time = decision_time + timedelta(days=forecast_horizon_days)
    explicit_time = _read_field(period, "realized_price_time")

    explicit_price = _finite_float(
        _first_present(period, "realized_price", "period_end_price", "actual_price")
    )
    if explicit_price is not None:
        if explicit_time is None and dates and not length_mismatch:
            fallback_time = dates[-1]
        return explicit_price, _safe_datetime(explicit_time, fallback_time)

    if not actual_prices:
        return None, _safe_datetime(explicit_time, fallback_time)

    realized_price = actual_prices[-1]
    if explicit_time is not None:
        realized_time = _safe_datetime(explicit_time, fallback_time)
    elif dates and not length_mismatch:
        realized_time = dates[-1]
    else:
        realized_time = fallback_time
    return realized_price, realized_time


def _resolve_defer_strategy_price(
    period: Any, actual_prices: List[float], realized_price: Optional[float]
) -> Optional[float]:
    mode = str(_read_field(period, "defer_price_mode", "period_end")).lower()
    if mode in {"average", "avg", "mean"} and actual_prices:
        return sum(actual_prices) / len(actual_prices)
    return realized_price


def _is_finite_price(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value) and value >= 0.0


def _calculate_commission(price: float, quantity: float, config: BacktestConfig) -> float:
    cost_config = config.costs
    gross_commission = (
        price * quantity * cost_config.commission_rate
        + quantity * cost_config.commission_per_unit
        + cost_config.fixed_fee
    )
    if cost_config.min_commission > 0:
        gross_commission = max(gross_commission, cost_config.min_commission)
    return max(0.0, gross_commission)


def _calculate_slippage(price: float, quantity: float, config: BacktestConfig) -> Tuple[float, float]:
    slippage_config = config.slippage
    if not slippage_config.apply_to_market_orders:
        return 0.0, 0.0
    per_unit = (
        price * (slippage_config.slippage_bps + slippage_config.spread_bps) / 10000.0
        + slippage_config.slippage_per_unit
    )
    per_unit = max(0.0, per_unit)
    return per_unit, per_unit * quantity


def _resolve_execution_bar_price(
    period: Any,
    dates: List[datetime],
    actual_prices: List[float],
    execution_time: Optional[datetime],
) -> Tuple[Optional[float], Optional[str]]:
    explicit_execution_price = _finite_float(_read_field(period, "execution_price"))
    if explicit_execution_price is not None:
        return explicit_execution_price, "explicit_execution_price"
    if execution_time is None:
        return None, None
    if not dates or not actual_prices or len(dates) != len(actual_prices):
        return None, None

    for date_index, date_value in enumerate(dates):
        if date_value == execution_time:
            return actual_prices[date_index], "execution_bar_price"
    return None, None


def _resolve_prices_for_action(
    period: Any,
    action: ProcurementAction,
    current_price: Optional[float],
    actual_prices: List[float],
    realized_price: Optional[float],
    dates: List[datetime],
    execution_time: Optional[datetime],
    config: BacktestConfig,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    explicit_baseline = _finite_float(_read_field(period, "baseline_price"))

    if action in {ProcurementAction.BUY_NOW, ProcurementAction.LOCK_CONTRACT}:
        if config.execution_timing == ExecutionTiming.NEXT_BAR_OPEN:
            strategy_price, price_source = _resolve_execution_bar_price(
                period, dates, actual_prices, execution_time
            )
        else:
            strategy_price = current_price
            price_source = "current_price"
        baseline_price = explicit_baseline if explicit_baseline is not None else realized_price
    elif action == ProcurementAction.DEFER:
        strategy_price = _resolve_defer_strategy_price(period, actual_prices, realized_price)
        price_source = "realized_price"
        baseline_price = explicit_baseline if explicit_baseline is not None else current_price
    else:
        fallback = explicit_baseline
        if fallback is None:
            fallback = realized_price if realized_price is not None else current_price
        strategy_price = fallback
        price_source = "non_executable_reference"
        baseline_price = fallback

    return strategy_price, baseline_price, price_source


def _create_order_and_fill(
    *,
    period_index: int,
    signal: Signal,
    strategy_price: Optional[float],
    baseline_price: Optional[float],
    execution_time: Optional[datetime],
    realized_price_time: datetime,
    price_source: Optional[str],
    config: BacktestConfig,
) -> Tuple[List[Order], List[Fill], Dict[str, float]]:
    executable_actions = {
        ProcurementAction.BUY_NOW,
        ProcurementAction.LOCK_CONTRACT,
        ProcurementAction.DEFER,
    }
    if signal.action not in executable_actions:
        return [], [], {
            "commission": 0.0,
            "slippage": 0.0,
            "slippage_cost": 0.0,
            "effective_strategy_price": strategy_price or 0.0,
            "realized_pnl": 0.0,
            "is_executable": False,
            "unavailable_reason": "non_executable_action",
        }

    if execution_time is None:
        return [], [], {
            "commission": 0.0,
            "slippage": 0.0,
            "slippage_cost": 0.0,
            "effective_strategy_price": strategy_price or 0.0,
            "realized_pnl": 0.0,
            "is_executable": False,
            "unavailable_reason": "missing_future_execution_time",
        }

    if not _is_finite_price(strategy_price):
        return [], [], {
            "commission": 0.0,
            "slippage": 0.0,
            "slippage_cost": 0.0,
            "effective_strategy_price": strategy_price or 0.0,
            "realized_pnl": 0.0,
            "is_executable": False,
            "unavailable_reason": "missing_execution_price",
        }

    quantity = float(signal.quantity or 0.0)
    if quantity <= 0:
        return [], [], {
            "commission": 0.0,
            "slippage": 0.0,
            "slippage_cost": 0.0,
            "effective_strategy_price": strategy_price,
            "realized_pnl": 0.0,
            "is_executable": False,
            "unavailable_reason": "invalid_quantity",
        }

    slippage, slippage_cost = _calculate_slippage(strategy_price, quantity, config)
    effective_price = strategy_price + slippage
    commission = _calculate_commission(effective_price, quantity, config)
    realized_pnl = 0.0
    if _is_finite_price(baseline_price):
        realized_pnl = (float(baseline_price) - effective_price) * quantity - commission

    order_id = f"order-{period_index}"
    order = Order(
        instrument=signal.instrument,
        quantity=quantity,
        decision_time=signal.decision_time,
        action=signal.action,
        execution_timing=config.execution_timing,
        order_id=order_id,
        signal_id=signal.signal_id,
        execution_time=execution_time,
        metadata={
            "price_source": price_source,
            "strategy_price": strategy_price,
            "baseline_price": baseline_price,
        },
    )
    fill = Fill(
        instrument=signal.instrument,
        quantity=quantity,
        price=effective_price,
        execution_time=execution_time,
        order_id=order_id,
        decision_time=signal.decision_time,
        realized_price_time=realized_price_time,
        commission=commission,
        slippage=slippage,
        slippage_cost=slippage_cost,
        realized_pnl=realized_pnl,
        metadata={
            "action": signal.action.value,
            "gross_strategy_price": strategy_price,
            "baseline_price": baseline_price,
        },
    )
    return [order], [fill], {
        "commission": commission,
        "slippage": slippage,
        "slippage_cost": slippage_cost,
        "effective_strategy_price": effective_price,
        "realized_pnl": realized_pnl,
        "is_executable": True,
        "unavailable_reason": None,
    }


def _period_to_signal(
    period: Any,
    *,
    config: BacktestConfig,
    signal_config: ForecastSignalConfig,
    instrument: Optional[str],
    quantity: Any,
    period_index: int,
) -> Tuple[Signal, Dict[str, Any]]:
    dates = _resolve_dates(period)
    actual_prices = _finite_prices(_read_field(period, "actual_prices"))
    current_price = _finite_float(_read_field(period, "current_price"))
    decision_time = _resolve_decision_time(period, dates, period_index)
    forecast_horizon_days = _resolve_forecast_horizon(period, actual_prices, dates)
    resolved_instrument = instrument or _read_field(period, "instrument", "commodity")
    resolved_quantity = quantity if quantity is not None else _read_field(period, "quantity")

    signal = forecast_to_signal(
        current_price=current_price,
        decision_time=decision_time,
        forecast_horizon_days=forecast_horizon_days,
        instrument=resolved_instrument,
        predicted_prices=_read_field(period, "predicted_prices"),
        p50=_first_present(period, "p50", "forecast_p50"),
        p10=_first_present(period, "p10", "forecast_p10"),
        p90=_first_present(period, "p90", "forecast_p90"),
        predicted_p50=_read_field(period, "predicted_p50"),
        predicted_p10=_read_field(period, "predicted_p10"),
        predicted_p90=_read_field(period, "predicted_p90"),
        qa_passed=_read_field(period, "qa_passed", True),
        qa_status=_read_field(period, "qa_status"),
        model_disagreement=_read_field(period, "model_disagreement"),
        model_disagreement_pct=_read_field(period, "model_disagreement_pct"),
        quantity=resolved_quantity,
        config=signal_config,
        metadata={
            "period_index": period_index,
            "execution_timing": config.execution_timing.value,
        },
    )

    context = {
        "dates": dates,
        "actual_prices": actual_prices,
        "current_price": current_price,
        "decision_time": decision_time,
        "forecast_horizon_days": forecast_horizon_days,
        "instrument": resolved_instrument,
        "length_mismatch": len(dates) != len(actual_prices),
    }
    return signal, context


def run_procurement_backtest_period(
    period: Any,
    *,
    config: Optional[Any] = None,
    signal_config: Optional[Any] = None,
    instrument: Optional[str] = None,
    quantity: Any = None,
    period_index: int = 0,
    cumulative_savings: float = 0.0,
) -> Dict[str, Any]:
    """Run one procurement decision period and return contract objects."""

    backtest_config = _coerce_backtest_config(config)
    forecast_config = _coerce_signal_config(signal_config)
    signal, context = _period_to_signal(
        period,
        config=backtest_config,
        signal_config=forecast_config,
        instrument=instrument,
        quantity=quantity,
        period_index=period_index,
    )

    realized_price, realized_price_time = _resolve_realized_price(
        period,
        context["actual_prices"],
        context["dates"],
        context["decision_time"],
        context["forecast_horizon_days"],
    )
    signal = signal.model_copy(update={"realized_price_time": realized_price_time})
    execution_time = _resolve_execution_time(
        period,
        context["decision_time"],
        context["dates"],
        signal.action,
        realized_price_time,
    )
    strategy_price, baseline_price, price_source = _resolve_prices_for_action(
        period,
        signal.action,
        context["current_price"],
        context["actual_prices"],
        realized_price,
        context["dates"],
        execution_time,
        backtest_config,
    )

    orders, fills, cost_details = _create_order_and_fill(
        period_index=period_index,
        signal=signal,
        strategy_price=strategy_price,
        baseline_price=baseline_price,
        execution_time=execution_time,
        realized_price_time=realized_price_time,
        price_source=price_source,
        config=backtest_config,
    )

    quantity_value = float(signal.quantity or forecast_config.default_quantity)
    net_strategy_price = cost_details["effective_strategy_price"]
    if fills and quantity_value > 0:
        net_strategy_price += cost_details["commission"] / quantity_value

    included_in_savings = (
        bool(fills)
        and _is_finite_price(net_strategy_price)
        and _is_finite_price(baseline_price)
        and quantity_value > 0
    )
    saved_amount = (
        (float(baseline_price) - float(net_strategy_price)) * quantity_value
        if included_in_savings
        else 0.0
    )
    saved_rate = (
        saved_amount / (float(baseline_price) * quantity_value)
        if included_in_savings and baseline_price and quantity_value
        else 0.0
    )

    equity_time = (
        realized_price_time
        if realized_price is not None
        else execution_time or signal.decision_time
    )
    equity_point = EquityPoint(
        time=equity_time,
        equity=backtest_config.initial_capital + cumulative_savings + saved_amount,
        cash=backtest_config.initial_capital + cumulative_savings + saved_amount,
        position_value=0.0,
        metadata={
            "period_index": period_index,
            "action": signal.action.value,
            "point_type": "period_result",
        },
    )

    period_result = {
        "period_index": period_index,
        "instrument": signal.instrument,
        "action": signal.action.value,
        "decision_time": signal.decision_time,
        "execution_time": execution_time,
        "execution_timing": ExecutionTiming.NEXT_BAR_OPEN.value,
        "realized_price_time": realized_price_time,
        "current_price": context["current_price"],
        "expected_price": signal.expected_price,
        "realized_price": realized_price,
        "strategy_price": strategy_price,
        "effective_strategy_price": cost_details["effective_strategy_price"],
        "net_strategy_price": net_strategy_price,
        "baseline_price": baseline_price,
        "price_source": price_source,
        "quantity": quantity_value,
        "saved_amount": saved_amount,
        "saved_rate": saved_rate,
        "commission": cost_details["commission"],
        "slippage": cost_details["slippage"],
        "slippage_cost": cost_details["slippage_cost"],
        "realized_pnl": cost_details["realized_pnl"],
        "is_executable": cost_details["is_executable"],
        "unavailable_reason": cost_details["unavailable_reason"],
        "included_in_savings": included_in_savings,
        "length_mismatch": context["length_mismatch"],
        "signal_reason": signal.metadata.get("reason"),
    }

    return {
        "signal": signal,
        "orders": orders,
        "fills": fills,
        "equity_point": equity_point,
        "period_result": period_result,
    }


def _apply_drawdowns(equity_curve: List[EquityPoint]) -> List[EquityPoint]:
    peak: Optional[float] = None
    updated: List[EquityPoint] = []
    for point in equity_curve:
        equity = float(point.equity)
        peak = equity if peak is None else max(peak, equity)
        drawdown = (peak - equity) / peak if peak and peak > 0 else 0.0
        updated.append(point.model_copy(update={"drawdown": drawdown}))
    return updated


def run_procurement_backtest(
    periods: Any,
    *,
    config: Optional[Any] = None,
    signal_config: Optional[Any] = None,
    instrument: Optional[str] = None,
    quantity: Any = None,
) -> ProcurementBacktestResult:
    """Run a lightweight procurement backtest over one or more periods."""

    backtest_config = _coerce_backtest_config(config)
    forecast_config = _coerce_signal_config(signal_config)
    period_items = _periods_from_input(periods)
    empty_savings = calculate_procurement_savings([], [], [])

    if not period_items:
        metrics = calculate_strategy_metrics([], fills=[], initial_capital=backtest_config.initial_capital)
        return ProcurementBacktestResult(
            config=backtest_config,
            metrics=metrics,
            procurement_metrics=empty_savings,
            procurement_savings=empty_savings,
            metadata={"period_count": 0},
        )

    first_dates = _resolve_dates(period_items[0])
    initial_time = _resolve_decision_time(period_items[0], first_dates, 0)
    equity_curve: List[EquityPoint] = [
        EquityPoint(
            time=initial_time,
            equity=backtest_config.initial_capital,
            cash=backtest_config.initial_capital,
            position_value=0.0,
            drawdown=0.0,
            metadata={"point_type": "initial"},
        )
    ]
    signals: List[Signal] = []
    orders: List[Order] = []
    fills: List[Fill] = []
    period_results: List[Dict[str, Any]] = []
    savings_strategy_prices: List[float] = []
    baseline_prices: List[float] = []
    quantities: List[float] = []
    cumulative_savings = 0.0

    for period_index, period in enumerate(period_items):
        period_output = run_procurement_backtest_period(
            period,
            config=backtest_config,
            signal_config=forecast_config,
            instrument=instrument,
            quantity=quantity,
            period_index=period_index,
            cumulative_savings=cumulative_savings,
        )
        signal = period_output["signal"]
        period_orders = period_output["orders"]
        period_fills = period_output["fills"]
        period_result = period_output["period_result"]

        signals.append(signal)
        orders.extend(period_orders)
        fills.extend(period_fills)
        period_results.append(period_result)

        if period_result["included_in_savings"]:
            savings_strategy_prices.append(float(period_result["net_strategy_price"]))
            baseline_prices.append(float(period_result["baseline_price"]))
            quantities.append(float(period_result["quantity"]))
            cumulative_savings += float(period_result["saved_amount"])

        equity_curve.append(
            period_output["equity_point"].model_copy(
                update={
                    "equity": backtest_config.initial_capital + cumulative_savings,
                    "cash": backtest_config.initial_capital + cumulative_savings,
                }
            )
        )

    equity_curve = _apply_drawdowns(equity_curve)
    procurement_savings = calculate_procurement_savings(
        savings_strategy_prices, baseline_prices, quantities
    )
    metrics = calculate_strategy_metrics(
        equity_curve, fills=fills, initial_capital=backtest_config.initial_capital
    )

    return ProcurementBacktestResult(
        config=backtest_config,
        signals=signals,
        orders=orders,
        fills=fills,
        equity_curve=equity_curve,
        metrics=metrics,
        procurement_metrics=procurement_savings,
        procurement_savings=procurement_savings,
        period_results=period_results,
        metadata={
            "period_count": len(period_items),
            "execution_timing": backtest_config.execution_timing.value,
            "savings_period_count": len(savings_strategy_prices),
        },
    )
