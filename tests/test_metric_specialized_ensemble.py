import unittest

from backend.ml.ensemble import build_metric_specialized_ensemble, resolve_direction_change_pct


class MetricSpecializedEnsembleTests(unittest.TestCase):
    def test_uses_best_model_for_price_direction_and_interval_separately(self):
        predictions = {
            "price_model": {
                "p50": [100.0, 102.0, 104.0],
                "p10": [98.0, 100.0, 102.0],
                "p90": [102.0, 104.0, 106.0],
            },
            "direction_model": {
                "p50": [100.0, 99.0, 98.0],
                "p10": [99.0, 98.0, 97.0],
                "p90": [101.0, 100.0, 99.0],
            },
            "interval_model": {
                "p50": [200.0, 202.0, 204.0],
                "p10": [190.0, 189.0, 188.0],
                "p90": [215.0, 218.0, 221.0],
            },
        }
        metrics = {
            "price_model": {"mape": 1.0, "directional_accuracy": 40.0, "coverage_rate": 50.0},
            "direction_model": {"mape": 3.0, "directional_accuracy": 80.0, "coverage_rate": 20.0},
            "interval_model": {"mape": 4.0, "directional_accuracy": 20.0, "coverage_rate": 90.0},
        }

        result = build_metric_specialized_ensemble(predictions, metrics, current_price=101.0)

        self.assertEqual(result["p50"], [100.0, 102.0, 104.0])
        self.assertEqual(result["best_per_metric"]["price_accuracy"], "price_model")
        self.assertEqual(result["best_per_metric"]["direction"], "direction_model")
        self.assertEqual(result["best_per_metric"]["coverage"], "interval_model")
        self.assertEqual(result["direction"], "下跌")
        self.assertEqual(result["direction_confidence"], 80.0)
        self.assertEqual(result["p10"], [90.0, 89.0, 88.0])
        self.assertEqual(result["p90"], [115.0, 118.0, 121.0])

    def test_selects_price_direction_and_interval_models_per_forecast_segment(self):
        predictions = {
            "short_price": {
                "p50": [100, 101, 102, 103, 104, 105],
                "p10": [99, 100, 101, 102, 103, 104],
                "p90": [101, 102, 103, 104, 105, 106],
            },
            "mid_price": {
                "p50": [200, 201, 202, 203, 204, 205],
                "p10": [198, 199, 200, 201, 202, 203],
                "p90": [202, 203, 204, 205, 206, 207],
            },
            "long_price": {
                "p50": [300, 301, 302, 303, 304, 305],
                "p10": [297, 298, 299, 300, 301, 302],
                "p90": [303, 304, 305, 306, 307, 308],
            },
            "wide_interval": {
                "p50": [400, 401, 402, 403, 404, 405],
                "p10": [390, 391, 392, 393, 394, 395],
                "p90": [420, 421, 422, 423, 424, 425],
            },
            "tight_interval": {
                "p50": [500, 501, 502, 503, 504, 505],
                "p10": [497, 498, 499, 500, 501, 502],
                "p90": [506, 507, 508, 509, 510, 511],
            },
        }
        global_metrics = {
            name: {"mape": 9.0, "directional_accuracy": 30.0, "coverage_rate": 20.0}
            for name in predictions
        }
        segment_metrics = {
            "1-2天": {
                "short_price": {"mape": 1.0, "directional_accuracy": 40.0, "coverage_rate": 20.0},
                "mid_price": {"mape": 3.0, "directional_accuracy": 95.0, "coverage_rate": 10.0},
                "wide_interval": {"mape": 4.0, "directional_accuracy": 20.0, "coverage_rate": 90.0},
            },
            "3-4天": {
                "short_price": {"mape": 5.0, "directional_accuracy": 30.0, "coverage_rate": 25.0},
                "mid_price": {"mape": 1.0, "directional_accuracy": 60.0, "coverage_rate": 35.0},
                "tight_interval": {"mape": 4.0, "directional_accuracy": 99.0, "coverage_rate": 80.0},
            },
            "5-6天": {
                "long_price": {"mape": 1.0, "directional_accuracy": 70.0, "coverage_rate": 40.0},
                "mid_price": {"mape": 2.0, "directional_accuracy": 98.0, "coverage_rate": 20.0},
                "wide_interval": {"mape": 4.0, "directional_accuracy": 20.0, "coverage_rate": 100.0},
            },
        }

        result = build_metric_specialized_ensemble(
            predictions,
            global_metrics,
            current_price=100.0,
            segment_metrics=segment_metrics,
            segments=[
                {"key": "1-2天", "start": 0, "end": 2},
                {"key": "3-4天", "start": 2, "end": 4},
                {"key": "5-6天", "start": 4, "end": 6},
            ],
        )

        self.assertEqual(result["p50"], [100.0, 101.0, 202.0, 203.0, 304.0, 305.0])
        self.assertEqual(result["p10"], [90.0, 91.0, 199.0, 200.0, 294.0, 295.0])
        self.assertEqual(result["p90"], [120.0, 121.0, 208.0, 209.0, 324.0, 325.0])
        self.assertEqual(result["best_per_segment"]["1-2天"]["price_accuracy"], "short_price")
        self.assertEqual(result["best_per_segment"]["1-2天"]["direction"], "mid_price")
        self.assertEqual(result["best_per_segment"]["1-2天"]["coverage"], "wide_interval")
        self.assertEqual(result["best_per_segment"]["3-4天"]["price_accuracy"], "mid_price")
        self.assertEqual(result["best_per_segment"]["3-4天"]["direction"], "tight_interval")
        self.assertEqual(result["best_per_segment"]["3-4天"]["coverage"], "tight_interval")
        self.assertEqual(result["best_per_segment"]["5-6天"]["price_accuracy"], "long_price")
        self.assertEqual(result["best_per_segment"]["5-6天"]["direction"], "mid_price")
        self.assertEqual(result["best_per_segment"]["5-6天"]["coverage"], "wide_interval")

    def test_llm_direction_override_keeps_change_pct_sign_consistent(self):
        direction, pct = resolve_direction_change_pct("上涨", -1.23)
        self.assertEqual(direction, "上涨")
        self.assertEqual(pct, 1.23)

        direction, pct = resolve_direction_change_pct("下跌", 0.8)
        self.assertEqual(direction, "下跌")
        self.assertEqual(pct, -0.8)

        direction, pct = resolve_direction_change_pct("震荡", 3.0)
        self.assertEqual(direction, "震荡")
        self.assertEqual(pct, 0.0)

    def test_coverage_model_prefers_target_coverage_with_narrower_interval(self):
        predictions = {
            "very_wide": {"p50": [100, 101], "p10": [80, 81], "p90": [125, 126]},
            "calibrated": {"p50": [100, 101], "p10": [92, 93], "p90": [110, 111]},
            "under": {"p50": [100, 101], "p10": [98, 99], "p90": [102, 103]},
        }
        metrics = {
            "very_wide": {"mape": 2.0, "directional_accuracy": 50.0, "coverage_rate": 100.0, "mean_interval_width_pct": 45.0},
            "calibrated": {"mape": 2.1, "directional_accuracy": 50.0, "coverage_rate": 82.0, "mean_interval_width_pct": 18.0},
            "under": {"mape": 1.0, "directional_accuracy": 50.0, "coverage_rate": 75.0, "mean_interval_width_pct": 4.0},
        }

        result = build_metric_specialized_ensemble(predictions, metrics, current_price=100.0)

        self.assertEqual(result["best_per_metric"]["coverage"], "calibrated")


if __name__ == "__main__":
    unittest.main()
