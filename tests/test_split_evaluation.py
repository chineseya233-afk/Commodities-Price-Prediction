import unittest
from datetime import date

import pandas as pd

from backend.ml.split_evaluation import select_train_test_by_date


class SplitEvaluationTests(unittest.TestCase):
    def test_selects_2024_2025_train_and_2026_jan_apr_test(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2023-12-29",
                "2024-01-02",
                "2025-12-31",
                "2026-01-02",
                "2026-04-30",
                "2026-05-01",
            ]),
            "price": [1, 2, 3, 4, 5, 6],
        })

        train, test = select_train_test_by_date(
            df,
            train_start=date(2024, 1, 1),
            train_end=date(2025, 12, 31),
            test_start=date(2026, 1, 1),
            test_end=date(2026, 4, 30),
        )

        self.assertEqual(train["price"].tolist(), [2, 3])
        self.assertEqual(test["price"].tolist(), [4, 5])


if __name__ == "__main__":
    unittest.main()
