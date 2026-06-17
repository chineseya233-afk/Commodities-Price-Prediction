import unittest
from datetime import date

from backend.utils.date_utils import (
    build_calendar_dates,
    build_calendar_price_history,
    build_visible_forecast_targets,
    training_start_date,
)


class PredictionDateTests(unittest.TestCase):
    def test_forecast_dates_include_today_and_weekend_days(self):
        dates = build_calendar_dates(date(2026, 5, 23), 5)

        self.assertEqual(
            dates,
            [
                date(2026, 5, 23),
                date(2026, 5, 24),
                date(2026, 5, 25),
                date(2026, 5, 26),
                date(2026, 5, 27),
            ],
        )

    def test_training_start_covers_full_2024_2025_window_for_2026_validation(self):
        self.assertEqual(training_start_date(date(2026, 5, 23)), date(2024, 1, 1))

    def test_price_history_fills_calendar_gap_before_today(self):
        rows = [
            {"date": "2026-05-22", "price": 6318.0, "high": 6320.0, "low": 6310.0},
        ]

        history = build_calendar_price_history(rows, end_date=date(2026, 5, 23), max_days=90)

        self.assertEqual([row["date"] for row in history], ["2026-05-22", "2026-05-23"])
        self.assertEqual(history[-1]["price"], 6318.0)
        self.assertEqual(history[-1]["high"], 6320.0)
        self.assertEqual(history[-1]["low"], 6310.0)

    def test_visible_forecast_targets_keep_model_offsets_when_data_lags(self):
        targets = build_visible_forecast_targets(
            data_end_date=date(2026, 5, 22),
            display_start_date=date(2026, 5, 24),
            horizon=5,
        )

        self.assertEqual(
            targets,
            [
                (1, date(2026, 5, 24)),
                (2, date(2026, 5, 25)),
                (3, date(2026, 5, 26)),
                (4, date(2026, 5, 27)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
