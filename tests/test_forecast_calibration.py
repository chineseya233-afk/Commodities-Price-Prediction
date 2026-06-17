import unittest

import numpy as np

from backend.ml.forecast_calibration import (
    apply_conformal_intervals,
    calculate_horizon_residual_quantiles,
    calibrate_forecast_volatility,
    evaluate_interval_coverage,
)

try:
    import pandas as pd
except ImportError:
    pd = None


class ForecastCalibrationTests(unittest.TestCase):
    def test_flat_forecast_gets_deterministic_historical_volatility_shape(self):
        history = np.array([
            6100, 6105, 6110, 6050, 6045, 6040, 6200, 6210, 6220, 6000,
            5995, 5985, 6150, 6160, 6170, 5900, 5910, 5925, 6080, 6090,
            6105, 5860, 5875, 5890, 6020, 6040, 6060, 5820, 5840, 5860,
        ], dtype=float)
        p50 = [5860.0] * 10
        p10 = [5800.0] * 10
        p90 = [5920.0] * 10

        result = calibrate_forecast_volatility(p50, p10, p90, history)

        self.assertEqual(len(result["p50"]), 10)
        self.assertNotEqual(result["p50"], p50)
        self.assertGreater(np.std(np.diff(result["p50"])), 12.0)
        self.assertEqual(result["p50"][0], p50[0])
        for low, mid, high in zip(result["p10"], result["p50"], result["p90"]):
            self.assertLess(low, mid)
            self.assertGreater(high, mid)

    def test_already_variable_forecast_is_not_overwritten(self):
        history = np.array([5800, 5860, 5780, 5900, 5750, 5880, 5760, 5920], dtype=float)
        p50 = [5800.0, 5865.0, 5785.0, 5910.0]
        p10 = [5700.0, 5765.0, 5685.0, 5810.0]
        p90 = [5900.0, 5965.0, 5885.0, 6010.0]

        result = calibrate_forecast_volatility(p50, p10, p90, history)

        self.assertEqual(result["p50"], p50)
        self.assertEqual(result["p10"], p10)
        self.assertEqual(result["p90"], p90)

    def test_residual_quantiles_are_grouped_by_horizon(self):
        actuals = [
            {"horizon": 1, "actual": 105.0},
            {"horizon": 1, "actual": 96.0},
            {"horizon": 2, "actual": 117.0},
            {"horizon": 2, "actual": 121.0},
            {"horizon": 2, "actual": np.nan},
        ]
        predictions = [
            {"horizon": 1, "p50": 100.0},
            {"horizon": 1, "p50": 100.0},
            {"horizon": 2, "p50": 120.0},
            {"horizon": 2, "p50": 120.0},
            {"horizon": 2, "p50": 120.0},
        ]

        result = calculate_horizon_residual_quantiles(actuals, predictions, quantile=1.0)

        self.assertEqual(result, {1: 5.0, 2: 3.0})

    def test_conformal_intervals_widen_without_changing_center(self):
        forecast_points = [
            {"horizon": 1, "p10": 98.0, "p50": 100.0, "p90": 102.0},
            {"horizon": 2, "p10": 197.0, "p50": 200.0, "p90": 203.0},
        ]

        result = apply_conformal_intervals(forecast_points, {1: 10.0, 2: 4.0})

        self.assertEqual([row["p50"] for row in result], [100.0, 200.0])
        self.assertEqual(result[0]["p10"], 90.0)
        self.assertEqual(result[0]["p90"], 110.0)
        self.assertEqual(result[1]["p10"], 196.0)
        self.assertEqual(result[1]["p90"], 204.0)
        for row in result:
            self.assertLessEqual(row["p10"], row["p50"])
            self.assertLessEqual(row["p50"], row["p90"])
        self.assertEqual(forecast_points[0]["p10"], 98.0)

    def test_coverage_gap_and_per_horizon_diagnostics(self):
        actuals = [
            {"horizon": 1, "actual": 95.0},
            {"horizon": 1, "actual": 105.0},
            {"horizon": 2, "actual": 130.0},
            {"horizon": 2, "actual": 200.0},
        ]
        forecast_points = [
            {"horizon": 1, "p10": 90.0, "p90": 110.0},
            {"horizon": 1, "p10": 90.0, "p90": 110.0},
            {"horizon": 2, "p10": 120.0, "p90": 140.0},
            {"horizon": 2, "p10": 120.0, "p90": 140.0},
        ]

        result = evaluate_interval_coverage(actuals, forecast_points, target_coverage=0.8)

        self.assertAlmostEqual(result["coverage_rate"], 0.75)
        self.assertAlmostEqual(result["coverage_gap"], 0.05)
        self.assertAlmostEqual(result["mean_interval_width"], 20.0)
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["per_horizon"][1]["coverage_rate"], 1.0)
        self.assertAlmostEqual(result["per_horizon"][1]["coverage_gap"], -0.2)
        self.assertAlmostEqual(result["per_horizon"][2]["coverage_rate"], 0.5)
        self.assertAlmostEqual(result["per_horizon"][2]["coverage_gap"], 0.3)

    def test_empty_and_missing_inputs_are_safe(self):
        self.assertEqual(calculate_horizon_residual_quantiles([], []), {})
        self.assertEqual(apply_conformal_intervals([], {}), [])

        diagnostics = evaluate_interval_coverage(
            [{"actual": np.nan}, {"not_actual": 10.0}],
            [{"p10": 1.0, "p90": 2.0}, {"p10": None, "p90": "bad"}],
        )

        self.assertEqual(diagnostics["coverage_rate"], 0.0)
        self.assertEqual(diagnostics["count"], 0)
        self.assertEqual(diagnostics["per_horizon"], {})

        repaired = apply_conformal_intervals(
            [{"horizon": 1, "p10": 110.0, "p50": 100.0, "p90": 90.0}],
            {1: 2.0},
        )
        self.assertLessEqual(repaired[0]["p10"], repaired[0]["p50"])
        self.assertLessEqual(repaired[0]["p50"], repaired[0]["p90"])

    @unittest.skipIf(pd is None, "pandas is not installed")
    def test_dataframe_inputs_are_supported(self):
        actuals = pd.DataFrame({"horizon": [1, 1, 2], "actual": [101.0, 99.0, 210.0]})
        predictions = pd.DataFrame({"horizon": [1, 1, 2], "p50": [100.0, 100.0, 200.0]})
        forecast_points = pd.DataFrame({
            "horizon": [1, 2],
            "p10": [99.5, 198.0],
            "p50": [100.0, 200.0],
            "p90": [100.5, 202.0],
        })

        quantiles = calculate_horizon_residual_quantiles(actuals, predictions, quantile=1.0)
        result = apply_conformal_intervals(forecast_points, quantiles)

        self.assertEqual(quantiles, {1: 1.0, 2: 10.0})
        self.assertEqual(result.loc[0, "p50"], 100.0)
        self.assertEqual(result.loc[1, "p10"], 190.0)
        self.assertEqual(result.loc[1, "p90"], 210.0)


if __name__ == "__main__":
    unittest.main()
