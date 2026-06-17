import unittest

import numpy as np

from backend.ml.segment_evaluation import evaluate_forecast_segments


class SegmentEvaluationTests(unittest.TestCase):
    def test_evaluates_each_model_inside_each_forecast_segment(self):
        actual = [100, 102, 104, 106]
        predictions = {
            "short_model": {
                "p50": [100, 102, 120, 121],
                "p10": [99, 101, 119, 120],
                "p90": [101, 103, 121, 122],
            },
            "mid_model": {
                "p50": [90, 91, 104, 106],
                "p10": [89, 90, 103, 105],
                "p90": [91, 92, 105, 107],
            },
        }

        result = evaluate_forecast_segments(
            actual,
            predictions,
            baseline_price=98,
            segments=[
                {"key": "1-2天", "start": 0, "end": 2},
                {"key": "3-4天", "start": 2, "end": 4},
            ],
        )

        self.assertIn("short_model", result["1-2天"])
        self.assertIn("mid_model", result["3-4天"])
        self.assertLess(result["1-2天"]["short_model"]["mape"], result["1-2天"]["mid_model"]["mape"])
        self.assertLess(result["3-4天"]["mid_model"]["mape"], result["3-4天"]["short_model"]["mape"])
        self.assertEqual(result["1-2天"]["short_model"]["coverage_rate"], 100.0)
        self.assertEqual(result["3-4天"]["mid_model"]["coverage_rate"], 100.0)

    def test_accepts_numpy_arrays_from_backtest_path(self):
        actual = np.array([100.0, 102.0])
        predictions = {
            "model": {
                "p50": np.array([100.0, 102.0]),
                "p10": np.array([99.0, 101.0]),
                "p90": np.array([101.0, 103.0]),
            }
        }

        result = evaluate_forecast_segments(
            actual,
            predictions,
            baseline_price=98.0,
            segments=[{"key": "1-2天", "start": 0, "end": 2}],
        )

        self.assertEqual(result["1-2天"]["model"]["coverage_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
