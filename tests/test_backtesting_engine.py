import unittest
from datetime import datetime

from backend.backtesting.engine import run_procurement_backtest
from backend.backtesting.prediction_adapter import ForecastSignalConfig, forecast_to_signal
from backend.backtesting.schemas import ExecutionTiming, ProcurementAction


class ForecastSignalAdapterTests(unittest.TestCase):
    def test_upward_forecast_produces_buy_or_lock_signal(self):
        signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            predicted_prices=[103.0],
            decision_time=datetime(2026, 1, 1, 9, 0),
        )

        self.assertEqual(signal.action, ProcurementAction.BUY_NOW)
        self.assertEqual(signal.reference_price, 100.0)
        self.assertEqual(signal.expected_price, 103.0)
        self.assertFalse(signal.metadata["uses_realized_prices"])

        lock_signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            p50=108.0,
            decision_time=datetime(2026, 1, 1, 9, 0),
        )

        self.assertEqual(lock_signal.action, ProcurementAction.LOCK_CONTRACT)

    def test_downward_forecast_produces_defer_signal(self):
        signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            p50=95.0,
            decision_time=datetime(2026, 1, 1, 9, 0),
        )

        self.assertEqual(signal.action, ProcurementAction.DEFER)
        self.assertEqual(signal.metadata["reason"], "forecast_down")

    def test_quality_or_uncertainty_produces_risk_actions(self):
        qa_signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            p50=110.0,
            decision_time=datetime(2026, 1, 1, 9, 0),
            qa_passed=False,
        )
        wide_signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            p50=101.0,
            p10=75.0,
            p90=130.0,
            decision_time=datetime(2026, 1, 1, 9, 0),
        )
        disagreement_signal = forecast_to_signal(
            instrument="copper",
            current_price=100.0,
            p50=101.0,
            decision_time=datetime(2026, 1, 1, 9, 0),
            model_disagreement_pct=0.2,
        )

        self.assertEqual(qa_signal.action, ProcurementAction.REQUEST_QUOTE)
        self.assertEqual(wide_signal.action, ProcurementAction.REQUEST_QUOTE)
        self.assertEqual(disagreement_signal.action, ProcurementAction.HEDGE_REVIEW)


class ProcurementBacktestEngineTests(unittest.TestCase):
    def test_next_bar_open_time_semantics_are_recorded(self):
        decision_time = datetime(2026, 1, 1, 9, 0)
        next_bar_time = datetime(2026, 1, 2, 9, 0)

        result = run_procurement_backtest(
            [
                {
                    "instrument": "copper",
                    "current_price": 100.0,
                    "predicted_prices": [103.0],
                    "actual_prices": [100.0, 110.0],
                    "dates": [decision_time, next_bar_time],
                    "decision_time": decision_time,
                    "quantity": 1.0,
                }
            ]
        )

        self.assertEqual(result.signals[0].decision_time, decision_time)
        self.assertEqual(result.orders[0].execution_timing, ExecutionTiming.NEXT_BAR_OPEN)
        self.assertEqual(result.orders[0].execution_time, next_bar_time)
        self.assertEqual(result.fills[0].execution_time, next_bar_time)
        self.assertEqual(result.fills[0].realized_price_time, next_bar_time)
        self.assertAlmostEqual(result.fills[0].price, 110.0)
        self.assertEqual(result.period_results[0]["execution_time"], next_bar_time)
        self.assertEqual(result.period_results[0]["realized_price_time"], next_bar_time)
        self.assertAlmostEqual(result.period_results[0]["strategy_price"], 110.0)
        self.assertEqual(result.period_results[0]["price_source"], "execution_bar_price")

    def test_no_next_bar_does_not_create_same_bar_fill(self):
        decision_time = datetime(2026, 1, 1, 9, 0)

        result = run_procurement_backtest(
            [
                {
                    "instrument": "copper",
                    "current_price": 100.0,
                    "predicted_prices": [103.0],
                    "actual_prices": [100.0],
                    "dates": [decision_time],
                    "decision_time": decision_time,
                    "quantity": 1.0,
                }
            ]
        )

        self.assertEqual(result.signals[0].action, ProcurementAction.BUY_NOW)
        self.assertEqual(result.orders, [])
        self.assertEqual(result.fills, [])
        self.assertIsNone(result.period_results[0]["execution_time"])
        self.assertFalse(result.period_results[0]["is_executable"])
        self.assertEqual(
            result.period_results[0]["unavailable_reason"],
            "missing_future_execution_time",
        )

    def test_explicit_execution_price_overrides_execution_bar_price(self):
        decision_time = datetime(2026, 1, 1, 9, 0)
        execution_time = datetime(2026, 1, 2, 9, 0)

        result = run_procurement_backtest(
            [
                {
                    "instrument": "copper",
                    "current_price": 100.0,
                    "predicted_prices": [103.0],
                    "actual_prices": [100.0, 105.0],
                    "dates": [decision_time, execution_time],
                    "decision_time": decision_time,
                    "execution_time": execution_time,
                    "execution_price": 107.0,
                    "quantity": 1.0,
                }
            ]
        )

        self.assertEqual(result.signals[0].action, ProcurementAction.BUY_NOW)
        self.assertEqual(result.orders[0].execution_time, execution_time)
        self.assertAlmostEqual(result.fills[0].price, 107.0)
        self.assertEqual(
            result.orders[0].metadata["price_source"],
            "explicit_execution_price",
        )
        self.assertEqual(
            result.period_results[0]["price_source"],
            "explicit_execution_price",
        )
        self.assertNotEqual(result.fills[0].price, 100.0)
        self.assertNotEqual(result.fills[0].price, 105.0)

    def test_procurement_savings_for_up_and_down_periods_are_reasonable(self):
        periods = [
            {
                "instrument": "copper",
                "current_price": 100.0,
                "predicted_prices": [112.0],
                    "actual_prices": [100.0, 105.0, 120.0],
                    "dates": [
                        datetime(2026, 1, 1, 9, 0),
                        datetime(2026, 1, 2, 9, 0),
                        datetime(2026, 1, 3, 9, 0),
                    ],
                    "decision_time": datetime(2026, 1, 1, 9, 0),
                    "quantity": 1.0,
                },
            {
                "instrument": "copper",
                "current_price": 100.0,
                "predicted_prices": [90.0],
                    "actual_prices": [100.0, 95.0, 90.0],
                    "dates": [
                        datetime(2026, 1, 3, 9, 0),
                        datetime(2026, 1, 4, 9, 0),
                        datetime(2026, 1, 5, 9, 0),
                    ],
                    "decision_time": datetime(2026, 1, 3, 9, 0),
                    "quantity": 1.0,
                },
        ]

        result = run_procurement_backtest(periods)

        self.assertEqual(result.signals[0].action, ProcurementAction.LOCK_CONTRACT)
        self.assertEqual(result.signals[1].action, ProcurementAction.DEFER)
        self.assertAlmostEqual(result.period_results[0]["saved_amount"], 15.0)
        self.assertAlmostEqual(result.period_results[1]["saved_amount"], 10.0)
        self.assertAlmostEqual(result.procurement_savings["saved_amount"], 25.0)
        self.assertAlmostEqual(result.procurement_savings["average_strategy_price"], 97.5)
        self.assertAlmostEqual(result.procurement_savings["average_baseline_price"], 110.0)
        self.assertEqual(result.metrics["total_trades"], 2)

    def test_empty_input_is_safe(self):
        result = run_procurement_backtest([])

        self.assertEqual(result.signals, [])
        self.assertEqual(result.orders, [])
        self.assertEqual(result.fills, [])
        self.assertEqual(result.equity_curve, [])
        self.assertEqual(result.metrics["total_trades"], 0)
        self.assertEqual(result.procurement_savings["saved_amount"], 0.0)

    def test_missing_prediction_and_length_mismatch_do_not_crash(self):
        result = run_procurement_backtest(
            {
                "instrument": "copper",
                "current_price": 100.0,
                "actual_prices": [100.0, 105.0],
                "dates": [],
                "decision_time": datetime(2026, 1, 1, 9, 0),
            },
            signal_config=ForecastSignalConfig(),
        )

        self.assertEqual(result.signals[0].action, ProcurementAction.HOLD)
        self.assertEqual(result.orders, [])
        self.assertTrue(result.period_results[0]["length_mismatch"])
        self.assertEqual(
            result.period_results[0]["realized_price_time"],
            datetime(2026, 1, 3, 9, 0),
        )


if __name__ == "__main__":
    unittest.main()
