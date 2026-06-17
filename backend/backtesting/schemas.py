"""
Serializable contracts for backtesting procurement strategies.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcurementAction(str, Enum):
    BUY_NOW = "buy_now"
    DEFER = "defer"
    LOCK_CONTRACT = "lock_contract"
    REQUEST_QUOTE = "request_quote"
    HEDGE_REVIEW = "hedge_review"
    HOLD = "hold"


class ExecutionTiming(str, Enum):
    NEXT_BAR_OPEN = "next_bar_open"


class CostConfig(BaseModel):
    commission_rate: float = Field(default=0.0, ge=0.0)
    commission_per_unit: float = Field(default=0.0, ge=0.0)
    fixed_fee: float = Field(default=0.0, ge=0.0)
    min_commission: float = Field(default=0.0, ge=0.0)


class SlippageConfig(BaseModel):
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_per_unit: float = Field(default=0.0, ge=0.0)
    spread_bps: float = Field(default=0.0, ge=0.0)
    apply_to_market_orders: bool = True


class RiskConfig(BaseModel):
    max_position_size: Optional[float] = Field(default=None, gt=0.0)
    max_notional: Optional[float] = Field(default=None, gt=0.0)
    max_drawdown_pct: Optional[float] = Field(default=None, gt=0.0)
    stop_loss_pct: Optional[float] = Field(default=None, gt=0.0)
    take_profit_pct: Optional[float] = Field(default=None, gt=0.0)


class WalkForwardConfig(BaseModel):
    method: str = "rolling"
    train_window: Optional[int] = Field(default=None, gt=0)
    initial_train_window: Optional[int] = Field(default=None, gt=0)
    test_window: int = Field(default=1, gt=0)
    step: int = Field(default=1, gt=0)
    embargo: int = Field(default=0, ge=0)


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=1_000_000.0, gt=0.0)
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    costs: CostConfig = Field(default_factory=CostConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    walk_forward: Optional[WalkForwardConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Signal(BaseModel):
    instrument: str
    decision_time: datetime
    action: ProcurementAction = ProcurementAction.HOLD
    forecast_horizon_days: int = Field(default=1, ge=0)
    realized_price_time: Optional[datetime] = None
    signal_id: Optional[str] = None
    expected_price: Optional[float] = None
    reference_price: Optional[float] = None
    quantity: Optional[float] = Field(default=None, gt=0.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    instrument: str
    quantity: float = Field(gt=0.0)
    decision_time: datetime
    action: ProcurementAction
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    order_id: Optional[str] = None
    signal_id: Optional[str] = None
    execution_time: Optional[datetime] = None
    limit_price: Optional[float] = Field(default=None, gt=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Fill(BaseModel):
    instrument: str
    quantity: float
    price: float = Field(ge=0.0)
    execution_time: datetime
    order_id: Optional[str] = None
    decision_time: Optional[datetime] = None
    realized_price_time: Optional[datetime] = None
    commission: float = Field(default=0.0, ge=0.0)
    slippage: float = 0.0
    slippage_cost: float = Field(default=0.0, ge=0.0)
    realized_pnl: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    instrument: str
    quantity: float
    average_price: float = Field(ge=0.0)
    as_of_time: datetime
    market_price: Optional[float] = Field(default=None, ge=0.0)
    market_price_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EquityPoint(BaseModel):
    time: datetime
    equity: float
    cash: Optional[float] = None
    position_value: Optional[float] = None
    drawdown: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    signals: List[Signal] = Field(default_factory=list)
    orders: List[Order] = Field(default_factory=list)
    fills: List[Fill] = Field(default_factory=list)
    positions: List[Position] = Field(default_factory=list)
    equity_curve: List[EquityPoint] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    procurement_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
