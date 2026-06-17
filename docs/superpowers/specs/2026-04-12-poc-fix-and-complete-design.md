# POC Fix & Completion Design Spec

## Context

The Commodities Price Prediction SaaS POC is partially built but has 36 identified bugs (4 critical, 18 major, 14 minor) that prevent it from functioning correctly. Additionally, several spec-mandated features are incomplete or missing. The data pipeline crashes on startup due to deprecated Pandas APIs, the TFT model never actually trains, the frontend sends unauthenticated API calls, and critical UI text is unreadable.

This spec covers systematically fixing all issues and completing the POC per the original development specification (`大宗商品采购价格预测 SaaS 系统开发规程 (POC阶段).md`).

## Approach: Bottom-Up Infrastructure First

Fix from the data layer upward so each layer is stable before building on it. Eight workstreams in dependency order:

```
WS1: Data Pipeline ─► WS2: ML Models ─► WS3: EIA/FRED APIs
                                              │
WS5: LLM Service ──► WS4: QA Engine ◄────────┘
                          │
                     WS6: Backend API ─► WS7: Frontend Fixes ─► WS8: Frontend Features
```

**Note on ordering**: WS5 (LLM Service) runs before WS4 (QA Engine) because QA Layer 2 depends on a working LLM client. WS5 is independent of the data/ML pipeline so it can be developed in parallel with WS1-WS3.

## Constraints & Environment

- **GPU**: NVIDIA RTX 2070 Laptop — TFT should train on CUDA
- **LLM**: Primary = Xiaomi MiMo (api.xiaomimimo.com), Fallback = DeepSeek (`DEEPSEEK_API_KEY` from local `.env`)
- **Data**: Keep simulator + add EIA and FRED real API integrations
- **Prediction horizon**: 30 days with 3-tier confidence (per spec)

---

## WS1: Data Pipeline Fixes

**Files**: `backend/ml/preprocessing.py`, `backend/ml/feature_engineering.py`, `backend/ml/tft_model.py`

| Issue | Severity | Fix |
|-------|----------|-----|
| `fillna(method="ffill")` crashes on Pandas 2.2 (`tft_model.py:126`) | CRITICAL | Replace with `.ffill().bfill().fillna(0)` |
| `isocalendar().week` type issues (`feature_engineering.py:80`) | MAJOR | Use `.values.astype(int)` for numpy conversion |
| `.dropna()` loses 20+ rows (`feature_engineering.py:68-70`) | MAJOR | Forward-fill feature columns, only drop rows where target is NaN |
| `is_month_start`/`is_month_end` forward compat (`feature_engineering.py:81-82`) | MINOR | Add try/except fallback with manual day comparison |

**Verification**: Start backend, check logs for zero `DeprecationWarning`, verify `GET /api/data/quality` shows higher `total_records`.

---

## WS2: ML Model Fixes

**Files**: `backend/ml/tft_model.py`, `backend/ml/baseline_models.py`, `backend/main.py`

| Issue | Severity | Fix |
|-------|----------|-----|
| TFT `load()` only loads config, not weights (`tft_model.py:370-382`) | CRITICAL | Save/load `training_dataset` alongside model; reconstruct architecture via `from_dataset()`, then load `state_dict` |
| `torch.save()` without `TFT_AVAILABLE` check (`tft_model.py:353`) | CRITICAL | Guard with `if TFT_AVAILABLE` |
| TFT never trained — always falls back (`main.py:~204`) | MAJOR | Call `prepare_data()` → `train()` → `predict()` with try/except fallback |
| XGBoost index -1 logic error (`baseline_models.py:193-195`) | MAJOR | Update all lag features during recursive prediction, not just `price_lag_1` |
| Feature columns may exclude engineered features (`baseline_models.py:143-149`) | MAJOR | Use `pd.api.types.is_numeric_dtype()` instead of string dtype matching |
| `interpret_output()` signature mismatch (`tft_model.py:335-337`) | MAJOR | Use `predict(dl, mode="raw", return_x=True)` then pass raw output to `interpret_output()` |
| Prophet dates returned but ignored (`baseline_models.py:79`) | MAJOR | Use `freq='B'` in `make_future_dataframe()` for business day alignment |
| Training dataset not persisted (`tft_model.py:134-169`) | MINOR | Included in the save/load fix above |

**Verification**: Check `nvidia-smi` during startup for GPU usage. `GET /api/predictions/all-models` returns non-empty arrays for all 4 models. XGBoost predictions show variance across horizon.

---

## WS3: Data Source Integration (EIA + FRED)

**New files**: `backend/data_providers/eia_provider.py`, `backend/data_providers/fred_provider.py`
**Modified files**: `backend/data_providers/__init__.py`, `backend/config.py`, `backend/main.py`, `.env`

### EIA Provider
- Endpoint: `https://api.eia.gov/v2/petroleum/pri/spt/data/`
- Series: Diesel retail prices + WTI crude spot
- Implements `DataProvider` abstract interface
- Graceful fallback to simulator if API unavailable
- Response caching to avoid rate limits
- **Note**: The `.env` currently has `EIA_API_KEY=demo`. A real EIA API key (free, from eia.gov) is needed for production data. The `demo` key returns limited sample data.

### FRED Provider
- Endpoint: `https://api.stlouisfed.org/fred/series/observations`
- Series: USD/CNY exchange rate (`DEXCHUS`), Brent crude (`DCOILBRENTEU`), Fed funds rate (`DFF`)
- Returns macro indicators merged as additional features

### Integration in `main.py`
- Try EIA real data first → fallback to simulator
- Optionally merge FRED macro indicators into feature DataFrame
- `GET /api/data/quality` reflects actual data source used

**Verification**: `GET /api/data/prices` returns data with `source: "eia"` when API is available. Disconnect internet — system falls back to simulator without crashing.

---

## WS4: QA Engine Completion

**Files**: `backend/services/qa_service.py`, `backend/main.py`

### Layer 1 (Hard Rules) — Already working
- Value range checks (±2σ)
- 7-day cumulative swing > ±15% alert
- Negative/zero prediction rejection
- Ordering: P10 ≤ P50 ≤ P90
- Confidence interval width: 0.5%-20%

### Layer 2 (LLM Soft Validation) — NEW
- `validate_with_llm()` method on `QAService`
- Builds market context from last 30 days of prices
- Calls LLM to assess multi-factor correlation reasonableness
- Only triggered when Layer 1 passes (don't waste LLM calls on already-failed predictions)
- Wired into `main.py` prediction pipeline after Layer 1

### Model Disagreement Risk
- Compare P50 direction across all 4 models
- If models disagree on up/down, generate "市场分歧风险" alert

**Verification**: Check dashboard `risk_alerts` for model-specific QA results. If models disagree, a disagreement alert appears.

---

## WS5: LLM Service Fixes

**Files**: `backend/services/llm_service.py`, `backend/config.py`, `.env`

### DeepSeek Fallback Chain
```
MiMo API → DeepSeek API → Mock/Static Report
```
- Add `fallback_client` with DeepSeek credentials
- Primary call fails → try fallback → fail → return mock report
- Add `.env` entries: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`

### 3-Dimensional Risk Research Report — NEW
- New method `generate_risk_report()` on `LLMService`
- Output format per spec section 4.4:
  - Dimension 1: Market Sentiment Risk (OPEC, exchange rates, seasonal demand)
  - Dimension 2: Model Prediction Risk (QA results in plain language)
  - Dimension 3: Data & Policy Risk (NDRC adjustment windows)
- New endpoint: `GET /api/reports/risk`

### LLM QA Validation Support
- New method `validate_with_llm()` called by QA engine (WS4)
- Returns `{passed: bool, reasoning: str}`

**Verification**: Test with valid MiMo key → verify report. Invalidate MiMo key → verify DeepSeek fallback. Invalidate both → verify mock report.

---

## WS6: Backend API Fixes

**Files**: `backend/main.py`, `backend/models/schemas.py`

### Error Isolation
- Wrap each model's prediction in individual try/except
- One model crashing does not block others
- Log which models succeeded/failed

### 30-Day Prediction with 3-Tier Confidence
- Propagate `PREDICTION_HORIZON=30` from config to all model constructors
- Label each prediction point with tier: `precise` (T+1~T+7), `standard` (T+8~T+14), `fuzzy` (T+15~T+30)
- Include tier labels in API response

### New Endpoint: Walk-Forward Backtest
- `GET /api/backtest/results`
- 90-day training window, 7-day rolling step
- Uses XGBoost (fastest to retrain) for backtest predictions
- Returns `{backtest_results: [{period_start, period_end, actual[], predicted[]}]}`

### New Endpoint: Risk Report
- `GET /api/reports/risk`
- Returns the 3-dimensional risk report from WS5

**Verification**: All 4 models produce predictions. `GET /api/predictions/latest?model=tft` returns 30 points with tier labels. `GET /api/backtest/results` returns real actual-vs-predicted data.

---

## WS7: Frontend Bug Fixes

**Files**: `frontend/app.js`, `frontend/styles.css`, `frontend/index.html`

| Issue | Severity | Fix |
|-------|----------|-----|
| `authHeaders()` never called (`app.js:47,60,71`) | CRITICAL | Add `{headers: authHeaders()}` to all fetch calls |
| Analysis card text unreadable (`styles.css:655-660`) | CRITICAL | Change `color` to `#0a2e24` (dark green on mint) |
| Button hover state unreadable (`styles.css:213-217`) | MAJOR | Use `background: rgba(60,255,208,0.75)` with black text |
| Chart crashes on null predictions (`app.js:111-113`) | MAJOR | Add null guards at function entry, show "loading" state |
| No error feedback for failed APIs (`app.js:47-78`) | MAJOR | Add `showToast()` helper with error messages |
| Chart resize listeners leak (`app.js:267,368,462,527`) | MINOR | Store chart instances globally, single resize handler |
| Inline style overrides on kickers (`index.html:117-118`) | MINOR | Replace with CSS classes (`card-kicker--inverted`) |
| Risk alert missing field guards (`app.js:548-567`) | MINOR | Add `?.` optional chaining and defaults |
| KPI direction not validated (`app.js:541`) | MINOR | Validate against allowed values, default to `'stable'` |
| Report null checks (`app.js:569-587`) | MINOR | Add `Array.isArray()` guards |

**Verification**: Login with any account — dashboard loads data. Analysis card text readable. Button hover visible. No console errors.

---

## WS8: Frontend Feature Completion

**Files**: `frontend/app.js`, `frontend/styles.css`, `frontend/index.html`

### Real Backtest Data (replacing fake `Math.sin` noise)
- Fetch from `GET /api/backtest/results`
- Render actual vs predicted in backtest chart
- Render real deviations in deviation table (or "—" if no historical prediction available)

### Role-Based Views
- Executive login → dashboard view default, de-technicalized KPIs:
  - "MAPE: 0.62%" → "预测可信度: 99.4%"
  - "方向准确率: 85%" → "涨跌预判: 8/10次正确"
  - "覆盖率: 92%" → "价格落入预测区间概率: 92%"
- Procurement login → procurement view default, full technical metrics
- Admin → all views accessible

### Loading States
- Skeleton placeholders during data fetch
- Toast notifications for errors ("数据加载失败")
- Status bar reflects connection state

### Data Refresh
- Auto-poll dashboard every 5 minutes
- Don't re-fetch LLM reports on auto-refresh (expensive)

### 30-Day Chart with 3-Tier Confidence Bands
- T+1~T+7: Solid line, narrow band, full opacity
- T+8~T+14: Dashed line, medium band, medium opacity
- T+15~T+30: Dotted line, wide band, low opacity
- Use ECharts `markArea` for tier shading

### Table Alignment
- Change deviation table from 14 → 30 records to match prediction horizon

**Verification**: Login as `executive` — KPIs show business language. Backtest chart shows real data. Loading skeletons appear during fetch. Auto-refresh fires at 5-minute interval.

---

## End-to-End Verification Checklist

### Backend
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `pip install -r backend/requirements.txt` succeeds
- [ ] `uvicorn backend.main:app` starts without errors
- [ ] Logs show "Starting TFT training..." (GPU active)
- [ ] `GET /api/health` — all fields `true`
- [ ] `GET /api/data/quality` — completeness ≥ 95%
- [ ] `GET /api/predictions/all-models` — 4 models with predictions
- [ ] `GET /api/backtest/results` — real walk-forward data
- [ ] `GET /api/reports/latest` — structured LLM report
- [ ] `GET /api/reports/risk` — 3-dimensional risk report

### Frontend
- [ ] Login as `executive` → dashboard with de-technicalized KPIs
- [ ] Login as `procurement` → procurement view by default
- [ ] Price chart shows 30-day prediction with 3-tier confidence
- [ ] Analysis card readable (dark text on mint)
- [ ] Backtest chart shows real data (not sine noise)
- [ ] Deviation table shows real numbers
- [ ] Loading skeletons appear during data fetch
- [ ] Error toast on backend disconnect
- [ ] Auto-refresh fires every 5 minutes

### Error Resilience
- [ ] Invalid MiMo key → DeepSeek fallback works
- [ ] Invalid both LLM keys → mock report generated
- [ ] Remove PyTorch → TFT falls back gracefully
- [ ] One model crashes → others still produce predictions
- [ ] Backend down → frontend shows error state

---

## Critical Files (Priority Order)

1. `backend/ml/tft_model.py` — Most bugs: deprecated API, broken save/load, never trains
2. `backend/main.py` — Orchestration: error isolation, TFT activation, new endpoints
3. `frontend/app.js` — Auth headers, chart crashes, fake data, role-based views, loading
4. `frontend/styles.css` — WCAG contrast failures, hover states
5. `backend/services/llm_service.py` — DeepSeek fallback, 3D risk report
6. `backend/ml/baseline_models.py` — XGBoost index bug, feature columns, Prophet dates
7. `backend/ml/feature_engineering.py` — NaN handling, isocalendar fix
8. `backend/data_providers/eia_provider.py` — NEW: real data integration
9. `backend/data_providers/fred_provider.py` — NEW: macro indicators
10. `backend/services/qa_service.py` — Layer 2 LLM validation
