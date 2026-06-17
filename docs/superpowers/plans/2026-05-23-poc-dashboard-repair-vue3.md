# POC Dashboard Repair Vue3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the POC commodity prediction dashboard so model metrics use real holdout behavior, news sentiment affects the ensemble forecast, and the frontend runs as a Vue 3 static app.

**Architecture:** Keep FastAPI as the single server and continue serving static frontend files from `/`. Backend fixes focus on causal model training, holdout evaluation, deterministic news sentiment fallback, and API-compatible dashboard responses. Frontend migrates from global imperative DOM mutation to Vue 3 component state while preserving the current API contract.

**Tech Stack:** FastAPI, pandas/numpy/scikit-learn/XGBoost/Prophet/PyTorch Forecasting, unittest, Vue 3 CDN build, Apache ECharts.

---

### Task 1: Backend Regression Tests

**Files:**
- Create: `tests/test_model_evaluation.py`
- Create: `tests/test_news_sentiment.py`
- Create: `tests/test_prediction_dates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_evaluation.py
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
```

```python
# tests/test_news_sentiment.py
import unittest

from backend.services.news_service import NewsSentimentService


class NewsSentimentTests(unittest.TestCase):
    def test_rule_based_sentiment_maps_bullish_news_to_positive_adjustment(self):
        service = NewsSentimentService(fetch_timeout=0.01)
        news = [
            {"title": "OPEC supply cut lifts crude prices", "summary": "diesel demand rises while inventories fall", "source": "test"}
        ]

        result = service.analyze_rule_based(news, current_price=7800.0)

        self.assertGreater(result["sentiment_score"], 0)
        self.assertGreater(result["price_adjustment_pct"], 0)
        self.assertEqual(result["news_items"][0]["level"], "中")

    def test_adjustment_curve_decays_across_horizon(self):
        service = NewsSentimentService()

        curve = service.build_adjustment_curve(0.01, horizon=5)

        self.assertEqual(len(curve), 5)
        self.assertGreater(curve[0], curve[-1])
```

```python
# tests/test_prediction_dates.py
import unittest
from datetime import date

from backend.main import _build_business_dates


class PredictionDateTests(unittest.TestCase):
    def test_business_dates_are_unique_when_starting_on_weekend(self):
        dates = _build_business_dates(date(2026, 5, 23), 5)

        self.assertEqual(dates, [
            date(2026, 5, 25),
            date(2026, 5, 26),
            date(2026, 5, 27),
            date(2026, 5, 28),
            date(2026, 5, 29),
        ])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: failures/import errors because the new helper methods and news service do not exist yet.

---

### Task 2: Causal Baselines And Metrics

**Files:**
- Modify: `backend/ml/baseline_models.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Implement `ModelEvaluator.directional_accuracy`**

Add a helper that compares each forecast step against the prior known baseline and treats stable predictions as correct only when the actual move is also within tolerance.

- [ ] **Step 2: Align XGBoost training target**

Add `_build_supervised_frame()` so XGBoost trains on features at time `t` and target `price[t+1]`, excluding same-day leakage.

- [ ] **Step 3: Update XGBoost recursive feature generation**

Append predicted future rows and recompute calendar, lag, rolling, technical, and price-change features before each next prediction.

- [ ] **Step 4: Replace the single mini-backtest direction formula**

Use holdout predictions and `ModelEvaluator.directional_accuracy(..., baseline_actual=last_train_price)` instead of the old weighted formula that made stable predictions show as 30%.

---

### Task 3: News Sentiment Integration

**Files:**
- Create: `backend/services/news_service.py`
- Modify: `backend/main.py`
- Modify: `backend/services/llm_service.py`

- [ ] **Step 1: Add news fetching with deterministic fallback**

Fetch public RSS headlines when available. If network or parsing fails, return POC fallback market events based on oil supply, demand, inventory, FX, and policy risks.

- [ ] **Step 2: Add LLM/rule sentiment analysis**

Use LLM JSON output when available; otherwise run keyword scoring. Normalize to `sentiment_score`, `price_adjustment_pct`, and structured `news_items`.

- [ ] **Step 3: Apply sentiment to ensemble**

Apply a decaying adjustment curve to ensemble P50/P10/P90 before final QA and report generation. Store the result in `_cache["news_sentiment"]`.

- [ ] **Step 4: Include news in API reports**

Pass news context into analysis/risk prompts and expose `news_sentiment` in `/api/dashboard/summary`.

---

### Task 4: Vue 3 Frontend Migration

**Files:**
- Replace: `frontend/index.html`
- Create: `frontend/src/main.js`
- Create: `frontend/src/styles.css`
- Keep or ignore: `frontend/app.js`, `frontend/styles.css` for rollback compatibility

- [ ] **Step 1: Create Vue shell**

Use Vue 3 global CDN and ECharts CDN, mount `#app`, and move login/app state into Vue reactivity.

- [ ] **Step 2: Componentize dashboard views**

Create component objects for login, nav/status, executive dashboard, procurement dashboard, charts, tables, QA, risk report, and chat.

- [ ] **Step 3: Preserve API contract**

Keep using `/api/auth/login`, `/api/dashboard/summary`, `/api/backtest/results`, and `/api/chat`.

- [ ] **Step 4: Fix frontend lifecycle bugs**

Dispose/recreate ECharts through Vue refs/watchers, avoid repeated refresh intervals, escape LLM text via Vue templates, and render responsive layouts without nested card structures.

---

### Task 5: Verification

**Files:**
- No new files unless diagnostics reveal a missing test.

- [ ] **Step 1: Run unit tests**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Start API**

Run: `.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`

Expected: API starts and logs `API ready!`.

- [ ] **Step 3: Check core endpoints**

Run health, dashboard, metrics, prediction, risk report, and backtest requests.

Expected: dashboard JSON includes `ensemble`, model metrics not fixed at artificial 30%, `news_sentiment` is present, and prediction dates are unique business days.

- [ ] **Step 4: Check frontend**

Open `http://127.0.0.1:8000`, log in with `executive/exec123`, verify dashboard renders; then switch to procurement and verify charts/tables/chat panel render without console-breaking errors.
