import unittest

import numpy as np

from backend.services.qa_service import QAEngine


class QAServiceTests(unittest.TestCase):
    def test_cumulative_change_uses_first_seven_days_only(self):
        qa = QAEngine()
        historical = np.array([100.0])
        p50 = np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 130.0])

        result = qa._check_cumulative_change(p50, historical)

        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
