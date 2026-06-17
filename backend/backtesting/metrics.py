"""
Metrics for strategy backtests and procurement outcomes.
"""

import math
import statistics
from datetime import date, datetime
from typing import Any, Iterable, List, Optional, Sequence


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _equity_value(point: Any) -> Optional[float]:
    if isinstance(point, (int, float)):
        return float(point)

    value = _read_field(point, "equity")
    if value is None:
        value = _read_field(point, "total_equity")
    if value is None:
        return None
    return float(value)


def _point_time(point: Any) -> Optional[Any]:
    value = _read_field(point, "time")
    if value is None:
        value = _read_field(point, "timestamp")
    return value


def _as_list(values: Optional[Iterable[Any]]) -> List[Any]:
    if values is None:
        return []
    return list(values)


def _empty_strategy_metrics() -> dict:
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "total_commission": 0.0,
        "slippage_cost": 0.0,
    }


def _safe_sum(items: Iterable[Any], field_name: str) -> float:
    total = 0.0
    for item in items:
        value = _read_field(item, field_name, 0.0)
        if value is not None:
            total += float(value)
    return total


def _calculate_returns(equity_values: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for previous, current in zip(equity_values, equity_values[1:]):
        if previous == 0:
            returns.append(0.0)
        else:
            returns.append((current - previous) / previous)
    return [value for value in returns if math.isfinite(value)]


def _calculate_max_drawdown(equity_values: Sequence[float]) -> float:
    if not equity_values:
        return 0.0

    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _elapsed_days(first_time: Any, last_time: Any) -> Optional[float]:
    if isinstance(first_time, datetime) and isinstance(last_time, datetime):
        return (last_time - first_time).total_seconds() / 86400.0
    if isinstance(first_time, date) and isinstance(last_time, date):
        return float((last_time - first_time).days)
    return None


def _calculate_annual_return(
    equity_values: Sequence[float],
    initial_capital: float,
    first_time: Any,
    last_time: Any,
) -> float:
    if len(equity_values) < 2 or initial_capital <= 0:
        return 0.0

    ending_capital = equity_values[-1]
    ratio = ending_capital / initial_capital
    if ratio <= 0:
        return 0.0

    elapsed_days = _elapsed_days(first_time, last_time)
    if elapsed_days is not None and elapsed_days > 0:
        return ratio ** (365.0 / elapsed_days) - 1.0

    periods = len(equity_values) - 1
    if periods <= 0:
        return 0.0
    return ratio ** (252.0 / periods) - 1.0


def _trade_pnls(fills: Sequence[Any]) -> List[float]:
    pnls: List[float] = []
    for fill in fills:
        value = _read_field(fill, "realized_pnl")
        if value is None:
            value = _read_field(fill, "pnl")
        if value is None:
            value = _read_field(fill, "profit")
        if value is not None:
            pnls.append(float(value))
    return pnls


def calculate_strategy_metrics(
    equity_curve: Optional[Iterable[Any]],
    fills: Optional[Iterable[Any]] = None,
    initial_capital: Optional[float] = None,
) -> dict:
    """Calculate common strategy metrics with safe empty-input behavior."""
    fill_items = _as_list(fills)
    metrics = _empty_strategy_metrics()
    metrics["total_trades"] = len(fill_items)
    metrics["total_commission"] = _safe_sum(fill_items, "commission")
    metrics["slippage_cost"] = _safe_sum(fill_items, "slippage_cost")

    trade_pnls = _trade_pnls(fill_items)
    if trade_pnls:
        wins = sum(1 for value in trade_pnls if value > 0)
        gross_profit = sum(value for value in trade_pnls if value > 0)
        gross_loss = sum(value for value in trade_pnls if value < 0)
        metrics["win_rate"] = wins / len(trade_pnls)
        metrics["profit_factor"] = (
            gross_profit / abs(gross_loss) if gross_loss != 0 else 0.0
        )

    equity_points = _as_list(equity_curve)
    equity_values = [
        value
        for value in (_equity_value(point) for point in equity_points)
        if value is not None and math.isfinite(value)
    ]
    if not equity_values:
        return metrics

    starting_capital = (
        float(initial_capital) if initial_capital is not None else equity_values[0]
    )
    if starting_capital > 0:
        metrics["total_return"] = (equity_values[-1] - starting_capital) / starting_capital

    first_time = _point_time(equity_points[0]) if equity_points else None
    last_time = _point_time(equity_points[-1]) if equity_points else None
    metrics["annual_return"] = _calculate_annual_return(
        equity_values, starting_capital, first_time, last_time
    )
    metrics["max_drawdown"] = _calculate_max_drawdown(equity_values)

    returns = _calculate_returns(equity_values)
    if len(returns) >= 2:
        return_std = statistics.stdev(returns)
        metrics["volatility"] = return_std * math.sqrt(252.0)
        if return_std > 0:
            metrics["sharpe_ratio"] = (
                statistics.mean(returns) / return_std * math.sqrt(252.0)
            )

    return metrics


def _float_list(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    return [float(value) for value in values]


def calculate_procurement_savings(
    strategy_prices: Optional[Iterable[Any]],
    baseline_prices: Optional[Iterable[Any]],
    quantities: Optional[Iterable[Any]],
) -> dict:
    """Calculate weighted procurement savings against a baseline."""
    strategy = _float_list(strategy_prices)
    baseline = _float_list(baseline_prices)
    qty = _float_list(quantities)

    if len(strategy) != len(baseline) or len(strategy) != len(qty):
        raise ValueError("strategy_prices, baseline_prices, and quantities must match")

    if not strategy:
        return {
            "saved_amount": 0.0,
            "saved_rate": 0.0,
            "average_strategy_price": 0.0,
            "average_baseline_price": 0.0,
        }

    strategy_spend = sum(price * quantity for price, quantity in zip(strategy, qty))
    baseline_spend = sum(price * quantity for price, quantity in zip(baseline, qty))
    saved_amount = baseline_spend - strategy_spend
    total_quantity = sum(qty)

    return {
        "saved_amount": saved_amount,
        "saved_rate": saved_amount / baseline_spend if baseline_spend else 0.0,
        "average_strategy_price": (
            strategy_spend / total_quantity if total_quantity else 0.0
        ),
        "average_baseline_price": (
            baseline_spend / total_quantity if total_quantity else 0.0
        ),
    }
