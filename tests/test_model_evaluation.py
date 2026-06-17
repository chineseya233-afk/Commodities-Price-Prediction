import unittest

import numpy as np
import pandas as pd

from backend.ml.baseline_models import ModelEvaluator, XGBoostForecaster


class ModelEvaluationTests(unittest.TestCase):
    def test_stable_forecast_does_not_get_artificial_thirty_percent(self):
        actual = np.array([101.0, 102.0, 103.0, 104.0])
        predicted = np.array([100.0, 100.0, 100.0, 100.0])

        score = ModelEvaluator.directional_accuracy(
            actual,
            predicted,
            baseline_actual=100.0,
            baseline_predicted=100.0,
            tolerance_pct=0.0,
        )

        self.assertEqual(score, 0.0)

    def test_directional_accuracy_uses_baseline_to_score_first_day(self):
        actual = np.array([101.0, 102.0, 101.0])
        predicted = np.array([101.5, 102.5, 101.5])

        score = ModelEvaluator.directional_accuracy(
            actual,
            predicted,
            baseline_actual=100.0,
            baseline_predicted=100.0,
            tolerance_pct=0.0,
        )

        self.assertEqual(score, 100.0)

    def test_default_directional_accuracy_counts_real_yuan_moves(self):
        actual = np.array([6282.31, 6292.51, 6282.09, 6295.45, 6294.05, 6310.37, 6318.33])
        predicted = np.array([6306.23, 6312.59, 6311.20, 6311.83, 6315.32, 6291.34, 6295.22])

        score = ModelEvaluator.directional_accuracy(
            actual,
            predicted,
            baseline_actual=6315.57,
            baseline_predicted=6315.57,
        )

        self.assertGreaterEqual(score, 60.0)

    def test_price_accuracy_is_reported_from_mape(self):
        actual = np.array([100.0, 102.0, 104.0])
        predicted = np.array([99.0, 103.0, 105.0])

        metrics = ModelEvaluator.evaluate(actual, predicted, "sample")

        self.assertIn("price_accuracy", metrics)
        self.assertAlmostEqual(metrics["price_accuracy"], 99.0194, places=4)

    def test_xgboost_supervised_frame_predicts_next_day_not_same_day(self):
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-01", periods=6),
                "price": [100, 101, 103, 106, 110, 115],
                "price_lag_1": [99, 100, 101, 103, 106, 110],
                "day_of_week": [3, 4, 0, 1, 2, 3],
            }
        )
        forecaster = XGBoostForecaster(prediction_horizon=3)

        X, y, feature_cols = forecaster._build_supervised_frame(df, "price", None)

        self.assertEqual(feature_cols, ["price_lag_1", "day_of_week"])
        self.assertEqual(len(X), 5)
        self.assertEqual(y.tolist(), [101, 103, 106, 110, 115])


if __name__ == "__main__":
    unittest.main()
