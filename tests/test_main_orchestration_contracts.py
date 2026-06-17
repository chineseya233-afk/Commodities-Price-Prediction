import asyncio
import re
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")


class MainOrchestrationContractTests(unittest.TestCase):
    def test_convert_numpy_serializes_report_contract_types(self):
        import numpy as np
        from pydantic import BaseModel

        import backend.main as main

        class ReportModel(BaseModel):
            name: str
            produced_on: date
            produced_at: datetime

        payload = main.convert_numpy({
            "model": ReportModel(
                name="diesel",
                produced_on=date(2026, 1, 2),
                produced_at=datetime(2026, 1, 2, 3, 4, 5),
            ),
            "date": date(2026, 1, 3),
            "datetime": datetime(2026, 1, 4, 5, 6, 7),
            "np_datetime64": np.datetime64("2026-01-05T06:07:08"),
        })

        self.assertEqual(payload["model"], {
            "name": "diesel",
            "produced_on": "2026-01-02",
            "produced_at": "2026-01-02T03:04:05",
        })
        self.assertEqual(payload["date"], "2026-01-03")
        self.assertEqual(payload["datetime"], "2026-01-04T05:06:07")
        self.assertEqual(payload["np_datetime64"], "2026-01-05T06:07:08")

    def test_llm_adjustment_is_advisory_not_direct_forecast_rewrite(self):
        self.assertNotIn("optimized_prices", MAIN_SOURCE)
        self.assertNotIn("opt_prices", MAIN_SOURCE)
        self.assertIn("llm_optimization_applied = False", MAIN_SOURCE)
        self.assertIn('"llm_optimization_applied": llm_optimization_applied', MAIN_SOURCE)
        self.assertIn('"llm_adjustment_proposal": None', MAIN_SOURCE)

        forbidden_rewrites = [
            r"ens_p50\s*\[[^\]]+\]\s*=\s*.*llm",
            r"ens_p50\s*\[[^\]]+\]\s*=\s*.*opt",
            r"ens_p10\s*\[[^\]]+\]\s*=\s*.*opt",
            r"ens_p90\s*\[[^\]]+\]\s*=\s*.*opt",
        ]
        for pattern in forbidden_rewrites:
            self.assertIsNone(re.search(pattern, MAIN_SOURCE, flags=re.IGNORECASE))

    def test_structured_report_endpoint_and_cache_contract_exist(self):
        self.assertIn("build_forecast_evidence_bundle", MAIN_SOURCE)
        self.assertIn("generate_structured_analysis_report", MAIN_SOURCE)
        self.assertIn('"forecast_evidence_bundle"', MAIN_SOURCE)
        self.assertIn('"structured_report"', MAIN_SOURCE)
        self.assertIn('"llm_adjustment_proposal"', MAIN_SOURCE)
        self.assertIn('@app.get("/api/reports/structured")', MAIN_SOURCE)
        self.assertIn('"evidence_bundle_summary"', MAIN_SOURCE)
        self.assertIn('"evidence_ids"', MAIN_SOURCE)
        self.assertIn('"adjustment_proposal"', MAIN_SOURCE)

    def test_backtest_results_include_procurement_strategy_contract(self):
        self.assertIn("run_procurement_backtest", MAIN_SOURCE)
        self.assertIn('"backtest_results": results', MAIN_SOURCE)
        self.assertIn('"procurement_backtest": procurement_backtest', MAIN_SOURCE)
        for field in (
            '"metrics"',
            '"procurement_savings"',
            '"period_results"',
            '"signals"',
            '"orders"',
            '"fills"',
            '"equity_curve"',
        ):
            self.assertIn(field, MAIN_SOURCE)

        self.assertIn('"decision_time": decision_time', MAIN_SOURCE)
        self.assertIn('"current_price": current_period_price', MAIN_SOURCE)
        self.assertIn('"actual_prices": rounded_actual', MAIN_SOURCE)
        self.assertIn('"dates": actual_dates', MAIN_SOURCE)
        self.assertIn('"uses_future_prices_for_signal": False', MAIN_SOURCE)

    def test_regenerate_report_refreshes_structured_report_and_returns_legacy_report(self):
        import pandas as pd

        import backend.main as main

        original_cache = main._cache.copy()
        structured_report = {
            "executive_summary": "structured report refreshed",
            "adjustment_proposal": None,
        }
        legacy_report = {
            "summary": "legacy analysis report",
            "recommendations": ["hold"],
        }

        async def fake_refresh_structured_report():
            main._cache["forecast_evidence_bundle"] = {"bundle": "refreshed"}
            main._cache["structured_report"] = structured_report
            return structured_report

        try:
            main._cache.clear()
            main._cache.update(original_cache)
            main._cache.update({
                "price_data": pd.DataFrame({"price": [7600.0, 7610.0, 7620.0]}),
                "report": {"summary": "stale report"},
                "forecast_evidence_bundle": None,
                "structured_report": None,
                "llm_adjustment_proposal": None,
            })

            with (
                patch.object(
                    main,
                    "_select_report_inputs",
                    return_value=({"p50": [7630.0]}, {"mape": 0.1}, {"score": 0.0}),
                ),
                patch.object(
                    main,
                    "_refresh_structured_report",
                    new=AsyncMock(side_effect=fake_refresh_structured_report),
                ) as refresh_mock,
                patch.object(main.llm_service, "generate_analysis_report", new=AsyncMock(return_value=legacy_report)) as report_mock,
            ):
                response = asyncio.run(main.regenerate_report(user={"role": "admin"}))

            refresh_mock.assert_awaited_once()
            report_mock.assert_awaited_once()
            self.assertEqual(response["status"], "regenerated")
            self.assertEqual(response["structured_report"], structured_report)
            self.assertEqual(response["report"], legacy_report)
            self.assertEqual(main._cache["forecast_evidence_bundle"], {"bundle": "refreshed"})
            self.assertEqual(main._cache["structured_report"], structured_report)
            self.assertEqual(main._cache["report"], legacy_report)
        finally:
            main._cache.clear()
            main._cache.update(original_cache)

    def test_backtest_results_passes_procurement_period_without_future_signal_prices(self):
        import numpy as np
        import pandas as pd

        import backend.main as main

        original_cache = main._cache.copy()
        prices = np.linspace(7500.0, 7596.0, 97)
        dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
        price_data = pd.DataFrame({"date": dates, "price": prices})
        featured_data = price_data.assign(momentum=np.arange(len(prices), dtype=float))
        captured = {}

        class FakeXGBoostForecaster:
            def __init__(self, prediction_horizon):
                captured["prediction_horizon"] = prediction_horizon

            def train_and_predict(self, train_df):
                captured["train_len"] = len(train_df)
                return {
                    "p50": [7610.0] * 7,
                    "p10": [7590.0] * 7,
                    "p90": [7630.0] * 7,
                }

        def fake_run_procurement_backtest(periods, *, instrument, quantity):
            captured["periods"] = periods
            captured["instrument"] = instrument
            captured["quantity"] = quantity
            return SimpleNamespace(
                metrics={"total_return": 0.0},
                procurement_savings={"total_savings": 0.0},
                period_results=[{"period": 0}],
                signals=[],
                orders=[],
                fills=[],
                equity_curve=[],
            )

        try:
            main._cache.clear()
            main._cache.update(original_cache)
            main._cache.update({
                "price_data": price_data,
                "featured_data": featured_data,
            })

            with (
                patch.object(main, "XGBoostForecaster", FakeXGBoostForecaster),
                patch.object(main, "run_procurement_backtest", fake_run_procurement_backtest),
            ):
                response = asyncio.run(main.get_backtest_results())

            self.assertIn("backtest_results", response)
            self.assertIn("procurement_backtest", response)
            self.assertEqual(captured["prediction_horizon"], 7)
            self.assertEqual(captured["train_len"], 90)
            self.assertEqual(captured["instrument"], "diesel_0")
            self.assertEqual(captured["quantity"], 1.0)

            train_end = len(prices) - 7
            [period] = captured["periods"]
            self.assertEqual(period["current_price"], float(prices[train_end - 1]))
            self.assertEqual(period["predicted_prices"], [7610.0] * 7)
            self.assertEqual(period["p50"], 7610.0)
            self.assertIs(period["metadata"]["uses_future_prices_for_signal"], False)
        finally:
            main._cache.clear()
            main._cache.update(original_cache)


if __name__ == "__main__":
    unittest.main()
