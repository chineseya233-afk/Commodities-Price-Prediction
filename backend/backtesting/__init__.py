"""
Backtesting foundation for procurement strategy evaluation.
"""

from .metrics import calculate_procurement_savings, calculate_strategy_metrics
from .engine import (
    ProcurementBacktestResult,
    run_procurement_backtest,
    run_procurement_backtest_period,
)
from .prediction_adapter import ForecastSignalConfig, forecast_to_signal
from .schemas import (
    BacktestConfig,
    BacktestResult,
    CostConfig,
    EquityPoint,
    ExecutionTiming,
    Fill,
    Order,
    Position,
    ProcurementAction,
    RiskConfig,
    Signal,
    SlippageConfig,
    WalkForwardConfig,
)
from .walk_forward import WalkForwardFold, expanding_splits, rolling_splits

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostConfig",
    "EquityPoint",
    "ExecutionTiming",
    "Fill",
    "ForecastSignalConfig",
    "Order",
    "Position",
    "ProcurementBacktestResult",
    "ProcurementAction",
    "RiskConfig",
    "Signal",
    "SlippageConfig",
    "WalkForwardConfig",
    "WalkForwardFold",
    "calculate_procurement_savings",
    "calculate_strategy_metrics",
    "expanding_splits",
    "forecast_to_signal",
    "rolling_splits",
    "run_procurement_backtest",
    "run_procurement_backtest_period",
]
