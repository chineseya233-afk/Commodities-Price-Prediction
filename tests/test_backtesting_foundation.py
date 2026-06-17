import unittest
from datetime import datetime

from backend.backtesting.metrics import (
    calculate_procurement_savings,
    calculate_strategy_metrics,
)
from backend.backtesting.schemas import (
    BacktestConfig,
    EquityPoint,
    ExecutionTiming,
    Fill,
    Order,
    ProcurementAction,
    Signal,
)
from backend.backtesting.walk_forward import expanding_splits, rolling_splits


class WalkForwardFoundationTests(unittest.TestCase):
    def test_rolling_splits_use_exclusive_bounds(self):
        folds = rolling_splits(
            data_length=12,
            train_window=4,
            test_window=2,
            step=3,
        )

        self.assertEqual(
            [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
            [(0, 4, 4, 6), (3, 7, 7, 9), (6, 10, 10, 12)],
        )

    def test_rolling_splits_apply_embargo_gap(self):
        folds = rolling_splits(
            data_length=12,
            train_window=4,
            test_window=2,
            step=3,
            embargo=1,
        )

        self.assertEqual(
            [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
            [(0, 4, 5, 7), (3, 7, 8, 10)],
        )
        self.assertEqual(folds[0].embargo_start, 4)
        self.assertEqual(folds[0].embargo_end, 5)

    def test_expanding_splits_grow_train_window(self):
        folds = expanding_splits(
            data_length=10,
            initial_train_window=4,
            test_window=2,
            step=2,
        )

        self.assertEqual(
            [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
            [(0, 4, 4, 6), (0, 6, 6, 8), (0, 8, 8, 10)],
        )

    def test_expanding_splits_apply_embargo_gap(self):
        folds = expanding_splits(
            data_length=10,
            initial_train_window=4,
            test_window=2,
            step=2,
            embargo=1,
        )

        self.assertEqual(
            [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
            [(0, 4, 5, 7), (0, 6, 7, 9)],
        )


class MetricsFoundationTests(unittest.TestCase):
    def test_strategy_metrics_calculate_returns_drawdown_and_costs(self):
        equity_curve = [
            EquityPoint(time=datetime(2026, 1, 1), equity=100.0),
            EquityPoint(time=datetime(2026, 1, 2), equity=120.0),
            EquityPoint(time=datetime(2026, 1, 3), equity=90.0),
            EquityPoint(time=datetime(2026, 1, 4), equity=150.0),
        ]
        fills = [
            Fill(
                instrument="copper",
                quantity=1.0,
                price=100.0,
                execution_time=datetime(2026, 1, 2),
                commission=2.0,
                slippage_cost=1.0,
                realized_pnl=10.0,
            ),
            Fill(
                instrument="copper",
                quantity=1.0,
                price=95.0,
                execution_time=datetime(2026, 1, 3),
                commission=3.0,
                slippage_cost=0.5,
                realized_pnl=-5.0,
            ),
        ]

        metrics = calculate_strategy_metrics(equity_curve, fills=fills)

        self.assertAlmostEqual(metrics["total_return"], 0.5)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.25)
        self.assertEqual(metrics["total_trades"], 2)
        self.assertAlmostEqual(metrics["total_commission"], 5.0)
        self.assertAlmostEqual(metrics["slippage_cost"], 1.5)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)
        self.assertAlmostEqual(metrics["profit_factor"], 2.0)
        self.assertGreater(metrics["volatility"], 0.0)

    def test_strategy_metrics_are_safe_for_empty_and_single_point_inputs(self):
        empty_metrics = calculate_strategy_metrics([])
        single_point_metrics = calculate_strategy_metrics([100.0])

        self.assertEqual(empty_metrics["total_return"], 0.0)
        self.assertEqual(empty_metrics["max_drawdown"], 0.0)
        self.assertEqual(empty_metrics["total_trades"], 0)
        self.assertEqual(single_point_metrics["annual_return"], 0.0)
        self.assertEqual(single_point_metrics["volatility"], 0.0)

    def test_procurement_savings_are_quantity_weighted(self):
        result = calculate_procurement_savings(
            strategy_prices=[90.0, 110.0],
            baseline_prices=[100.0, 120.0],
            quantities=[2.0, 3.0],
        )

        self.assertAlmostEqual(result["saved_amount"], 50.0)
        self.assertAlmostEqual(result["saved_rate"], 50.0 / 560.0)
        self.assertAlmostEqual(result["average_strategy_price"], 102.0)
        self.assertAlmostEqual(result["average_baseline_price"], 112.0)

    def test_procurement_savings_are_safe_for_zero_baseline(self):
        result = calculate_procurement_savings([0.0], [0.0], [10.0])

        self.assertEqual(result["saved_amount"], 0.0)
        self.assertEqual(result["saved_rate"], 0.0)


class SchemaFoundationTests(unittest.TestCase):
    def test_schema_defaults_and_enum_values(self):
        decision_time = datetime(2026, 1, 1, 9, 0)

        config = BacktestConfig()
        signal = Signal(
            instrument="copper",
            decision_time=decision_time,
            forecast_horizon_days=7,
        )
        order = Order(
            instrument="copper",
            quantity=10.0,
            decision_time=decision_time,
            action=ProcurementAction.BUY_NOW,
        )

        self.assertEqual(ProcurementAction.BUY_NOW.value, "buy_now")
        self.assertEqual(ProcurementAction.DEFER.value, "defer")
        self.assertEqual(ProcurementAction.LOCK_CONTRACT.value, "lock_contract")
        self.assertEqual(ProcurementAction.REQUEST_QUOTE.value, "request_quote")
        self.assertEqual(ProcurementAction.HEDGE_REVIEW.value, "hedge_review")
        self.assertEqual(ProcurementAction.HOLD.value, "hold")
        self.assertEqual(config.execution_timing, ExecutionTiming.NEXT_BAR_OPEN)
        self.assertEqual(signal.action, ProcurementAction.HOLD)
        self.assertEqual(signal.forecast_horizon_days, 7)
        self.assertEqual(order.execution_timing, ExecutionTiming.NEXT_BAR_OPEN)
        self.assertEqual(signal.model_dump(mode="json")["decision_time"], "2026-01-01T09:00:00")


if __name__ == "__main__":
    unittest.main()
