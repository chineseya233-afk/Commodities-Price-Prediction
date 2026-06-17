"""
FastAPI Main Application (Backend Agent)

The central hub connecting data providers, ML models, QA engine, and LLM service.
Serves the complete REST API for the frontend dashboard.
"""

import asyncio
import os
import sys
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Any, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from loguru import logger

# passlib<1.7.5 reads bcrypt.__about__.__version__, which bcrypt 4.x removed.
# Provide the attribute to avoid a noisy trapped traceback during startup.
try:
    import bcrypt as _bcrypt

    if not hasattr(_bcrypt, "__about__"):
        class _BcryptAbout:
            __version__ = getattr(_bcrypt, "__version__", "unknown")

        _bcrypt.__about__ = _BcryptAbout()
except Exception:
    pass

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.data_providers.simulator import ChinaDieselSimulator
from backend.data_providers.eia_provider import EIAProvider
from backend.data_providers.fred_provider import FREDProvider
from backend.ml.preprocessing import DataPreprocessor
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.tft_model import TFTModel
from backend.ml.baseline_models import (
    NaiveForecaster, ProphetForecaster, XGBoostForecaster, ModelEvaluator
)
from backend.ml.ensemble import build_metric_specialized_ensemble
from backend.ml.forecast_calibration import calibrate_forecast_volatility
from backend.ml.segment_evaluation import DEFAULT_SEGMENTS as FORECAST_SEGMENTS
from backend.ml.segment_evaluation import evaluate_forecast_segments
from backend.ml.split_evaluation import select_train_test_by_date
from backend.backtesting import run_procurement_backtest
from backend.services.qa_service import QAEngine
from backend.services.llm_service import LLMService
from backend.services.news_service import NewsSentimentService
from backend.services.report_context_service import build_forecast_evidence_bundle, collect_evidence_ids
from backend.models.db_models import get_engine, create_tables, get_session, User
from backend.models.schemas import (
    LoginRequest, LoginResponse, UserInfo,
    PriceHistoryResponse, PricePoint, LatestPriceResponse,
    DataQualityResponse,
    ForecastEvidenceBundle,
)
from backend.utils.date_utils import (
    build_calendar_dates as _build_forecast_dates,
    build_calendar_price_history as _build_calendar_price_history,
    build_visible_forecast_targets as _build_visible_forecast_targets,
    training_start_date as _training_start_date,
)


# === Global State ===
simulator = ChinaDieselSimulator(seed=42)
eia_provider = EIAProvider(
    api_key=settings.eia_api_key,
    base_url=settings.eia_base_url,
    diesel_series=settings.eia_diesel_series,
)
fred_provider = FREDProvider(api_key=settings.fred_api_key, base_url=settings.fred_base_url)
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
tft_model = TFTModel(prediction_horizon=settings.prediction_horizon)
naive_forecaster = NaiveForecaster(prediction_horizon=settings.prediction_horizon)
prophet_forecaster = ProphetForecaster(prediction_horizon=settings.prediction_horizon)
xgboost_forecaster = XGBoostForecaster(prediction_horizon=settings.prediction_horizon)
qa_engine = QAEngine()
llm_service = LLMService(
    api_key=settings.resolved_llm_api_key,
    base_url=settings.resolved_llm_base_url,
    model=settings.resolved_llm_model,
)
news_sentiment_service = NewsSentimentService()
model_evaluator = ModelEvaluator()

# Cache for generated data and predictions
_cache: Dict = {
    "price_data": None,
    "featured_data": None,
    "predictions": {},
    "metrics": {},
    "report": None,
    "structured_report": None,
    "forecast_evidence_bundle": None,
    "llm_adjustment_proposal": None,
    "risk_report": None,
    "news_sentiment": None,
    "data_quality": None,
    "data_source": "simulator",
    "fixed_split_evaluation": None,
    "price_signature": None,
    "last_refresh_at": None,
    "last_refresh_checked_at": None,
    "news_refresh_at": None,
    "auto_repairs": {},
}
_refresh_lock = asyncio.Lock()
_report_refresh_task: asyncio.Task | None = None

# Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

DEFAULT_USERS = [
    {
        "username": "admin",
        "password": settings.admin_default_password,
        "previous_passwords": ["admin123", "Admin@2026#POC!"],
        "role": "admin",
        "full_name": "系统管理员",
    },
    {
        "username": "executive",
        "password": settings.executive_default_password,
        "previous_passwords": ["exec123", "Exec@2026#POC!"],
        "role": "executive",
        "full_name": "高管用户",
    },
    {
        "username": "procurement",
        "password": settings.procurement_default_password,
        "previous_passwords": ["proc123", "Buy@2026#POC!"],
        "role": "procurement",
        "full_name": "采购专员",
    },
]


# === Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting Commodity Price Prediction API...")

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Initialize database
    engine = get_engine("sqlite:///./data/commodity_prediction.db")
    create_tables(engine)

    # Seed default users and rotate the old weak POC passwords if this database
    # was created before cookie-based auth.
    session = get_session(engine)
    if session.query(User).count() == 0:
        users = [
            User(username=item["username"], hashed_password=pwd_context.hash(item["password"]),
                 role=item["role"], full_name=item["full_name"])
            for item in DEFAULT_USERS
        ]
        for u in users:
            session.add(u)
        session.commit()
        logger.info("Default users seeded: admin/executive/procurement")
    else:
        rotated = []
        for item in DEFAULT_USERS:
            user = session.query(User).filter(User.username == item["username"]).first()
            if not user:
                continue
            if pwd_context.verify(item["password"], user.hashed_password):
                continue
            for previous_password in item.get("previous_passwords", []):
                if pwd_context.verify(previous_password, user.hashed_password):
                    user.hashed_password = pwd_context.hash(item["password"])
                    user.role = item["role"]
                    user.full_name = item["full_name"]
                    rotated.append(item["username"])
                    break
        if rotated:
            session.commit()
            logger.warning(f"Rotated weak POC default passwords for: {', '.join(rotated)}")
    session.close()

    # Pre-generate simulation data
    await _initialize_data()
    refresh_task = asyncio.create_task(_market_refresh_loop())

    logger.info("API ready!")
    try:
        yield
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        logger.info("Shutting down...")


app = FastAPI(
    title="大宗商品价格预测 SaaS API",
    description="Commodity Price Prediction SaaS - POC (柴油)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if (
        request.url.path.startswith("/api/")
        or request.url.path == "/"
        or request.url.path.startswith("/src/")
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


# === Numpy Conversion Helper ===
def convert_numpy(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        if hasattr(obj, "model_dump"):
            try:
                return convert_numpy(obj.model_dump(mode="json"))
            except TypeError:
                return convert_numpy(obj.model_dump())
        return convert_numpy(obj.dict())
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        try:
            timestamp = pd.to_datetime(obj)
            if pd.isna(timestamp):
                return None
            return timestamp.isoformat()
        except (TypeError, ValueError):
            return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(convert_numpy(k)): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    return obj


# === Helper Functions ===
def _price_signature(price_df: pd.DataFrame) -> dict | None:
    """Return a compact signature for detecting upstream price changes."""
    if price_df is None or price_df.empty or "date" not in price_df.columns or "price" not in price_df.columns:
        return None
    latest = price_df.sort_values("date").iloc[-1]
    return {
        "date": str(pd.to_datetime(latest["date"]).date()),
        "price": round(float(latest["price"]), 4),
        "records": int(len(price_df)),
    }


def _news_refresh_due() -> bool:
    refreshed_at = _cache.get("news_refresh_at")
    if not refreshed_at:
        return True
    try:
        age_seconds = (datetime.now() - datetime.fromisoformat(str(refreshed_at))).total_seconds()
    except ValueError:
        return True
    return age_seconds >= max(int(getattr(settings, "data_refresh_seconds", 300)), 300)


async def _load_market_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Fetch and process the latest market dataset without mutating cache."""
    end_date = date.today()
    try:
        start_date = date.fromisoformat(str(settings.market_data_start_date))
    except ValueError:
        start_date = _training_start_date(end_date)
    data_source = "simulator"

    # Try EIA real data first
    price_df = pd.DataFrame()
    try:
        price_df = await eia_provider.fetch_price_data("diesel_0", start_date, end_date)
        if not price_df.empty and len(price_df) >= 30:
            data_source = "eia"
            logger.info(f"Using EIA real data: {len(price_df)} records")
    except Exception as e:
        logger.warning(f"EIA unavailable: {e}")

    # Fall back to simulator
    if price_df.empty or len(price_df) < 30:
        price_df = await simulator.fetch_price_data("diesel_0", start_date, end_date)
        data_source = "simulator"
        logger.info(f"Using simulator data: {len(price_df)} records")

    # Try merging FRED macro indicators as extra features
    try:
        macro_df = await fred_provider.fetch_macro_indicators(start_date, end_date)
        if not macro_df.empty:
            price_df["date"] = pd.to_datetime(price_df["date"])
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            price_df = pd.merge(price_df, macro_df, on="date", how="left")
            price_df = price_df.ffill().bfill()
            if "price_usd_gallon" in price_df.columns and "dexchus" in price_df.columns:
                usd_gallon = pd.to_numeric(price_df["price_usd_gallon"], errors="coerce")
                usd_cny = pd.to_numeric(price_df["dexchus"], errors="coerce").ffill().bfill()
                converted_price = usd_gallon * EIAProvider.GALLONS_PER_TON * usd_cny
                valid_conversion = converted_price.notna() & (converted_price > 0)
                price_df.loc[valid_conversion, "price"] = converted_price[valid_conversion].round(2)
                price_df.loc[valid_conversion, "open"] = (converted_price[valid_conversion] * 0.998).round(2)
                price_df.loc[valid_conversion, "high"] = (converted_price[valid_conversion] * 1.005).round(2)
                price_df.loc[valid_conversion, "low"] = (converted_price[valid_conversion] * 0.995).round(2)
                logger.info("EIA diesel prices converted with FRED USD/CNY (DEXCHUS)")
            logger.info(f"FRED macro indicators merged: {list(macro_df.columns)}")
    except Exception as e:
        logger.warning(f"FRED unavailable: {e}")

    # Preprocess
    price_df = preprocessor.process(price_df, target_col="price")
    data_quality = preprocessor.get_quality_report()

    # Feature engineering
    featured_df = feature_engineer.create_features(price_df, target_col="price")
    return price_df, featured_df, data_quality, data_source


async def _refresh_data_cache(force_rebuild: bool = False) -> bool:
    """Refresh source data and rebuild predictions only when prices changed."""
    async with _refresh_lock:
        if (
            not force_rebuild
            and _cache.get("data_source") == "simulator"
            and _cache.get("price_signature")
        ):
            _cache["last_refresh_checked_at"] = datetime.now().isoformat()
            logger.info("Simulator data is stable within the current backend session; refresh skipped")
            return False

        price_df, featured_df, data_quality, data_source = await _load_market_dataset()
        new_signature = _price_signature(price_df)
        _cache["last_refresh_checked_at"] = datetime.now().isoformat()

        if not force_rebuild and new_signature == _cache.get("price_signature"):
            if _news_refresh_due():
                logger.info("Market data unchanged, rebuilding forecasts for due news refresh")
                _cache["predictions"] = {}
                _cache["metrics"] = {}
                _cache["auto_repairs"] = {}
                await _generate_all_predictions()
                _cache["fixed_split_evaluation"] = _evaluate_fixed_train_test_split()
                _cache["last_refresh_at"] = datetime.now().isoformat()
                return True
            logger.info(f"Market data unchanged: {new_signature}")
            return False

        _cache["predictions"] = {}
        _cache["metrics"] = {}
        _cache["report"] = None
        _cache["structured_report"] = None
        _cache["forecast_evidence_bundle"] = None
        _cache["llm_adjustment_proposal"] = None
        _cache["risk_report"] = None
        _cache["news_sentiment"] = None
        _cache["data_quality"] = data_quality
        _cache["data_source"] = data_source
        _cache["price_signature"] = new_signature
        _cache["last_refresh_at"] = datetime.now().isoformat()
        _cache["auto_repairs"] = {}

        _cache["price_data"] = price_df
        _cache["featured_data"] = featured_df

        logger.info(
            f"Data ready: {len(price_df)} price records, {len(featured_df)} featured records "
            f"(source: {data_source}, signature: {new_signature})"
        )

        await _generate_all_predictions()
        _cache["fixed_split_evaluation"] = _evaluate_fixed_train_test_split()
        return True


async def _initialize_data():
    """Generate and cache data on startup, then keep cache refreshable."""
    logger.info("Initializing data...")
    await _refresh_data_cache(force_rebuild=True)


async def _market_refresh_loop():
    """Refresh market data in the background while the backend stays alive."""
    interval = max(int(getattr(settings, "data_refresh_seconds", 300)), 60)
    while True:
        await asyncio.sleep(interval)
        try:
            changed = await _refresh_data_cache(force_rebuild=False)
            if changed:
                logger.info("Background market refresh rebuilt predictions")
        except Exception as e:
            logger.warning(f"Background market refresh failed: {e}")


async def _refresh_data_if_requested(refresh: bool = False):
    if refresh:
        await _refresh_data_cache(force_rebuild=False)


def _build_default_analysis_report(
    current_price: float,
    predictions: dict,
    model_metrics: dict,
    qa_summary: str,
    news_sentiment: dict | None,
) -> dict:
    p50 = [float(v) for v in predictions.get("p50", []) if isinstance(v, (int, float))]
    p10 = [float(v) for v in predictions.get("p10", []) if isinstance(v, (int, float))]
    p90 = [float(v) for v in predictions.get("p90", []) if isinstance(v, (int, float))]
    horizon = min(7, len(p50))
    short_prices = p50[:horizon] or [current_price]
    low = min((p10[:horizon] or short_prices))
    high = max((p90[:horizon] or short_prices))
    avg = sum(short_prices) / len(short_prices)
    end_price = short_prices[-1]
    trend = "下行" if end_price < current_price else "上行" if end_price > current_price else "震荡"
    action = "逢低分批采购" if trend == "下行" else "控制追高采购" if trend == "上行" else "维持常规采购"
    reasoning = (
        f"当前价格约{current_price:.0f}元/吨，未来7天预测均价约{avg:.0f}元/吨，"
        f"预测区间约{low:.0f}-{high:.0f}元/吨。系统已先基于模型与规则生成可执行建议，"
        "后台模型报告刷新完成后会自动更新更细的研判文本。"
    )
    risk_factors = []
    if news_sentiment and news_sentiment.get("summary"):
        risk_factors.append(str(news_sentiment.get("summary"))[:80])
    if qa_summary and "全部通过" not in qa_summary:
        risk_factors.append(str(qa_summary)[:80])

    return {
        "summary": f"综合模型预测短端价格偏{trend}，建议{action}。",
        "trend_analysis": reasoning,
        "risk_factors": risk_factors,
        "procurement_advice": {
            "action": action,
            "confidence": "中",
            "reasoning": reasoning,
            "suggested_price_range": f"{low:.0f}-{high:.0f} RMB/吨",
            "timing": f"未来7天接近{low:.0f}元/吨附近优先分批执行",
        },
        "data_quality_notes": f"数据源: {_cache.get('data_source', 'unknown')}; 模型指标: {model_metrics.get('model', 'ensemble')}",
    }


def _build_default_risk_report(qa_failures: dict, news_sentiment: dict | None) -> dict:
    market_items = []
    if news_sentiment and news_sentiment.get("summary"):
        market_items.append({
            "level": "关注",
            "title": str(news_sentiment.get("summary"))[:80],
            "impact": "新闻和舆情因子已纳入短端价格调整，建议继续跟踪后续变化。",
        })

    model_items = [
        {
            "level": "关注",
            "title": f"{name} 模型QA提示",
            "impact": str(info.get("summary") or "该模型未通过单项质量校验，已从核心决策模型选择中剔除或降权。"),
        }
        for name, info in (qa_failures or {}).items()
    ]

    return {
        "report_date": str(date.today()),
        "dimension_1_market": market_items,
        "dimension_2_model": model_items,
        "dimension_3_policy": [],
    }


def _failed_qa_checks(checks: list[dict]) -> list[dict]:
    return [
        {"check": item.get("check"), "detail": item.get("detail", "")}
        for item in checks or []
        if not item.get("passed", True)
    ]


def _safe_prediction_baseline(historical: np.ndarray, horizon: int) -> dict:
    base = float(historical[-1]) if len(historical) else 7800.0
    p50 = [round(base, 2)] * horizon
    return {
        "p50": p50,
        "p10": [round(base * 0.985, 2)] * horizon,
        "p90": [round(base * 1.015, 2)] * horizon,
        "mean": p50,
        "model": "safe_baseline",
    }


def _repair_prediction_guardrails(pred: dict, historical: np.ndarray, horizon: int) -> dict:
    base = float(historical[-1]) if len(historical) else 7800.0
    p50 = [float(v) for v in pred.get("p50", [])[:horizon]]
    if len(p50) < horizon:
        p50.extend([p50[-1] if p50 else base] * (horizon - len(p50)))

    repaired_p50 = []
    for i, value in enumerate(p50):
        prev = base if i == 0 else repaired_p50[-1]
        value = min(max(value, prev * 0.95), prev * 1.05)
        if i < 7:
            value = min(max(value, base * 0.851), base * 1.149)
        value = min(max(value, 3000.0), 15000.0)
        repaired_p50.append(round(value, 2))

    repaired_p10 = []
    repaired_p90 = []
    for i, value in enumerate(repaired_p50):
        spread = value * (0.015 if i < 7 else 0.035 if i < 14 else 0.06)
        repaired_p10.append(round(max(1.0, value - spread), 2))
        repaired_p90.append(round(value + spread, 2))

    repaired = dict(pred)
    repaired.update({"p50": repaired_p50, "p10": repaired_p10, "p90": repaired_p90, "mean": repaired_p50})
    return repaired


def _ai_repair_plan(model_name: str, summary: str, failed_checks: list[dict]) -> dict | None:
    if not llm_service.is_available():
        return None
    prompt = {
        "model_name": model_name,
        "qa_summary": summary,
        "failed_checks": failed_checks,
        "task": (
            "Return JSON with root_cause, repair_steps, and monitoring_checks for fixing "
            "a commodity price forecasting model. Do not propose direct forecast values."
        ),
    }
    try:
        raw = llm_service._call_llm(
            "You are a senior ML reliability engineer. Return JSON only.",
            str(prompt),
            temperature=0.1,
            max_tokens=700,
        )
        parsed = llm_service._parse_llm_response(raw) if raw else None
        return parsed if isinstance(parsed, dict) else {"raw": raw}
    except Exception as exc:
        logger.warning(f"AI repair plan failed for {model_name}: {exc}")
        return {"error": str(exc)}


def _attempt_prediction_auto_repair(
    model_name: str,
    pred: dict,
    historical: np.ndarray,
    summary: str,
    qa_checks: list[dict],
) -> tuple[dict, dict]:
    failed_checks = _failed_qa_checks(qa_checks)
    horizon = len(pred.get("p50", [])) or settings.prediction_horizon
    repaired = _repair_prediction_guardrails(pred, historical, horizon)
    repaired_valid, repaired_summary, repaired_details = qa_engine.validate_predictions(
        repaired, historical, f"{model_name}:auto_repair"
    )
    repair_record = {
        "model": model_name,
        "trigger": "qa_failure",
        "original_summary": summary,
        "failed_checks": failed_checks,
        "status": "repaired" if repaired_valid else "needs_manual_review",
        "repair_strategy": "deterministic_guardrails",
        "repaired_summary": repaired_summary,
    }
    if not repaired_valid:
        repair_record["ai_plan"] = _ai_repair_plan(model_name, summary, failed_checks)
        return pred, repair_record

    repaired["qa_passed"] = True
    repaired["qa_summary"] = repaired_summary
    repaired["qa_checks"] = repaired_details.get("layer1_checks", [])
    repaired["qa_auto_repaired"] = True
    repaired["qa_original_summary"] = summary
    repaired["qa_original_checks"] = qa_checks
    repaired["repair_strategy"] = "deterministic_guardrails"
    return repaired, repair_record


def _record_model_runtime_failure(model_name: str, exc: Exception):
    summary = f"{model_name} runtime failure: {exc}"
    failed_checks = [{"check": "runtime_error", "detail": str(exc)}]
    _cache.setdefault("auto_repairs", {})[model_name] = {
        "model": model_name,
        "trigger": "runtime_error",
        "status": "fallback_or_excluded",
        "original_summary": summary,
        "failed_checks": failed_checks,
        "ai_plan": _ai_repair_plan(model_name, summary, failed_checks),
    }


async def _refresh_llm_reports_background(
    current_price: float,
    ensemble_pred: dict,
    historical_prices: list,
    ens_metrics: dict,
    ens_summary: str,
    news_sentiment: dict,
    qa_failures: dict,
):
    await asyncio.sleep(1)
    try:
        evidence_bundle = build_forecast_evidence_bundle(_cache, commodity="diesel_0")
        _cache["forecast_evidence_bundle"] = evidence_bundle
        structured_report = await asyncio.to_thread(
            lambda: asyncio.run(llm_service.generate_structured_analysis_report(evidence_bundle))
        )
        _cache["structured_report"] = structured_report
        proposal = getattr(structured_report, "adjustment_proposal", None)
        proposal_payload = convert_numpy(proposal) if proposal else None
        _cache["llm_adjustment_proposal"] = proposal_payload
        ensemble = _cache.get("predictions", {}).get("ensemble")
        if isinstance(ensemble, dict):
            ensemble["llm_adjustment_proposal"] = proposal_payload
    except Exception as exc:
        logger.warning(f"Background structured report refresh failed: {exc}")

    try:
        _cache["report"] = await asyncio.to_thread(
            lambda: asyncio.run(llm_service.generate_analysis_report(
                current_price=current_price,
                predictions=ensemble_pred,
                historical_prices=historical_prices[-30:],
                model_metrics=ens_metrics,
                qa_summary=ens_summary,
                news_sentiment=news_sentiment,
            ))
        )
    except Exception as exc:
        logger.warning(f"Background analysis report refresh failed: {exc}")

    try:
        _cache["risk_report"] = await asyncio.to_thread(
            lambda: asyncio.run(llm_service.generate_risk_report(
                current_price=current_price,
                predictions=ensemble_pred,
                qa_results=qa_failures if qa_failures else {"all": {"passed": True, "summary": "全部通过"}},
                model_metrics=ens_metrics,
                news_sentiment=news_sentiment,
            ))
        )
    except Exception as exc:
        logger.warning(f"Background risk report refresh failed: {exc}")


def _schedule_llm_report_refresh(
    current_price: float,
    ensemble_pred: dict,
    historical_prices: list,
    ens_metrics: dict,
    ens_summary: str,
    news_sentiment: dict,
    qa_failures: dict,
) -> None:
    global _report_refresh_task
    if _report_refresh_task and not _report_refresh_task.done():
        _report_refresh_task.cancel()
    _report_refresh_task = asyncio.create_task(_refresh_llm_reports_background(
        current_price,
        ensemble_pred,
        historical_prices,
        ens_metrics,
        ens_summary,
        news_sentiment,
        qa_failures,
    ))


async def _generate_all_predictions():
    """Generate predictions from all models with error isolation, then run QA."""
    df = _cache.get("featured_data")
    price_df = _cache.get("price_data")

    if df is None or price_df is None:
        return

    historical = price_df["price"].values

    # 1. Naive Forecast
    try:
        naive_pred = naive_forecaster.predict(df)
        _cache["predictions"]["naive"] = naive_pred
        logger.info("Naive forecast complete")
    except Exception as e:
        logger.error(f"Naive forecast failed: {e}")
        _record_model_runtime_failure("naive", e)

    # 2. Prophet
    try:
        prophet_pred = prophet_forecaster.train_and_predict(price_df)
        _cache["predictions"]["prophet"] = prophet_pred
        logger.info("Prophet forecast complete")
    except Exception as e:
        logger.error(f"Prophet failed: {e}")
        _record_model_runtime_failure("prophet", e)

    # 3. XGBoost
    try:
        xgb_pred = xgboost_forecaster.train_and_predict(df)
        _cache["predictions"]["xgboost"] = xgb_pred
        logger.info("XGBoost forecast complete")
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        _record_model_runtime_failure("xgboost", e)

    # 4. TFT (load portable artifact first, train only when needed)
    try:
        model_path = os.path.join(settings.trained_models_dir, "tft_model.pt")
        if os.path.exists(model_path):
            tft_model.load(model_path)
            if tft_model.is_trained:
                tft_pred = tft_model.predict(data=df)
                tft_pred["model"] = "TFT"
                logger.info(f"TFT loaded from portable artifact: {model_path}")
            else:
                raise RuntimeError("TFT artifact exists but could not be loaded")
        else:
            training_data, val_data, train_dl, val_dl = tft_model.prepare_data(df)
            if train_dl is not None:
                logger.info("Starting TFT training on GPU...")
                train_result = tft_model.train(train_dl, val_dl, training_data)
                if train_result.get("status") == "completed":
                    tft_pred = tft_model.predict(data=df)
                    tft_pred["model"] = "TFT"
                    logger.info(f"TFT training completed in {train_result.get('epochs', '?')} epochs")
                    tft_model.save(model_path)
                else:
                    tft_pred = tft_model._fallback_predict(df)
                    tft_pred["model"] = "TFT (fallback)"
                    logger.warning(f"TFT training incomplete: {train_result}")
            else:
                tft_pred = tft_model._fallback_predict(df)
                tft_pred["model"] = "TFT (fallback)"
                logger.info("TFT using fallback (PyTorch Forecasting unavailable)")
    except Exception as e:
        logger.warning(f"TFT load/train failed: {e}")
        _record_model_runtime_failure("tft", e)
        try:
            training_data, val_data, train_dl, val_dl = tft_model.prepare_data(df)
        except Exception:
            training_data, train_dl, val_dl = None, None, None
        if train_dl is not None:
            logger.info("Starting TFT training on GPU...")
            train_result = tft_model.train(train_dl, val_dl, training_data)
            if train_result.get("status") == "completed":
                tft_pred = tft_model.predict(data=df)
                tft_pred["model"] = "TFT"
                logger.info(f"TFT training completed in {train_result.get('epochs', '?')} epochs")
                tft_model.save(model_path)
            else:
                tft_pred = tft_model._fallback_predict(df)
                tft_pred["model"] = "TFT (fallback)"
                logger.warning(f"TFT training incomplete: {train_result}")
        else:
            tft_pred = tft_model._fallback_predict(df)
            tft_pred["model"] = "TFT (fallback)"
            logger.info("TFT using fallback (PyTorch Forecasting unavailable)")
    _cache["predictions"]["tft"] = tft_pred

    # === QA Validation ===
    for model_name, pred in _cache["predictions"].items():
        # Sanitize any NaN/Inf in predictions before QA
        for key in ("p50", "p10", "p90", "mean"):
            if key in pred:
                sanitized = []
                for v in pred[key]:
                    v = float(v)
                    if np.isnan(v) or np.isinf(v) or v <= 0:
                        v = float(historical[-1]) if len(historical) > 0 else 7800.0
                    sanitized.append(round(v, 2))
                pred[key] = sanitized

        # Layer 1: Hard rules
        is_valid, summary, details = qa_engine.validate_predictions(
            pred, historical, model_name
        )
        pred["qa_passed"] = is_valid
        pred["qa_summary"] = summary
        pred["qa_checks"] = details.get("layer1_checks", [])
        if not is_valid:
            repaired_pred, repair_record = _attempt_prediction_auto_repair(
                model_name, pred, historical, summary, pred["qa_checks"]
            )
            _cache.setdefault("auto_repairs", {})[model_name] = repair_record
            pred.update(repaired_pred)
            is_valid = bool(pred.get("qa_passed", False))
        pred["excluded_from_ensemble"] = not is_valid

        # Layer 2 LLM validation is advisory and can be slow. Keep startup
        # deterministic; background LLM reports provide the softer analysis.
        pred["qa_l2_status"] = "deferred"

    # Model disagreement detection
    p50_directions = {}
    for model_name, pred in _cache["predictions"].items():
        p50 = pred.get("p50", [])
        if len(p50) >= 2:
            p50_directions[model_name] = "up" if p50[-1] > p50[0] else "down"
    _cache["model_disagreement"] = len(set(p50_directions.values())) > 1

    # === REAL MINI-BACKTEST for metrics ===
    # Pretend today is 7 business days ago: train only on data before holdout.
    model_weights = {"tft": 0.40, "prophet": 0.30, "xgboost": 0.20, "naive": 0.10}
    test_len = min(7, max(len(historical) - 31, 0))
    if test_len <= 0:
        test_len = min(7, max(len(historical) - 1, 0))

    actual_test = historical[-test_len:].copy() if test_len > 0 else np.array([])
    baseline_price = float(historical[-test_len - 1]) if test_len > 0 and len(historical) > test_len else float(historical[-1])
    backtest_price_df = price_df.iloc[:-test_len].copy() if test_len > 0 else price_df.copy()
    backtest_df = feature_engineer.create_features(backtest_price_df, target_col="price") if len(backtest_price_df) else df.copy()

    logger.info(f"Mini-backtest: using {len(backtest_df)} rows for training, {test_len} days for validation")

    backtest_outputs = {}

    if test_len > 0:
        try:
            bt_naive = NaiveForecaster(prediction_horizon=test_len).predict(backtest_df)
            backtest_outputs["naive"] = bt_naive
        except Exception as e:
            logger.warning(f"Naive backtest failed: {e}")
            backtest_outputs["naive"] = {
                "p50": [baseline_price] * test_len,
                "p10": [baseline_price * 0.99] * test_len,
                "p90": [baseline_price * 1.01] * test_len,
            }

        try:
            bt_prophet = ProphetForecaster(prediction_horizon=test_len).train_and_predict(backtest_price_df)
            backtest_outputs["prophet"] = bt_prophet
        except Exception as e:
            logger.warning(f"Prophet backtest failed: {e}")

        try:
            bt_xgb = XGBoostForecaster(prediction_horizon=test_len).train_and_predict(backtest_df)
            backtest_outputs["xgboost"] = bt_xgb
        except Exception as e:
            logger.warning(f"XGBoost backtest failed: {e}")

        try:
            # Do not evaluate the already trained full-data TFT on holdout data.
            # For POC speed, use the TFT wrapper's statistical fallback on train-only data.
            bt_tft = TFTModel(prediction_horizon=test_len)._fallback_predict(backtest_df)
            bt_tft["model"] = "TFT backtest fallback"
            backtest_outputs["tft"] = bt_tft
        except Exception as e:
            logger.warning(f"TFT backtest fallback failed: {e}")

    if test_len > 0:
        train_history_for_interval = backtest_price_df["price"].astype(float).values if not backtest_price_df.empty else historical[:-test_len]
        for model_name, bt_pred in list(backtest_outputs.items()):
            if len(bt_pred.get("p50", [])) >= test_len:
                backtest_outputs[model_name] = _apply_historical_interval_floor(bt_pred, train_history_for_interval)

    for model_name, bt_pred in backtest_outputs.items():
        bt_slice = bt_pred.get("p50", [])[:test_len]
        if len(bt_slice) < test_len or test_len <= 0:
            continue

        metrics = model_evaluator.evaluate(actual_test, bt_slice, model_name)
        metrics["directional_accuracy"] = model_evaluator.directional_accuracy(
            actual_test,
            bt_slice,
            baseline_actual=baseline_price,
            baseline_predicted=baseline_price,
        )

        bt_p10 = bt_pred.get("p10", bt_slice)[:test_len]
        bt_p90 = bt_pred.get("p90", bt_slice)[:test_len]
        if len(bt_p10) >= test_len and len(bt_p90) >= test_len:
            metrics["coverage_rate"] = model_evaluator.coverage_rate(actual_test, bt_p10, bt_p90)
            metrics["interval_calibrated"] = bool(bt_pred.get("interval_calibrated"))

        if model_name == "naive" and float(np.var(bt_slice)) <= 0.1:
            metrics["directional_accuracy_applicable"] = False
            metrics["metric_notes"] = {
                "directional_accuracy": "naive_flat_baseline",
                "directional_accuracy_label": "平线基线不参与方向评分",
            }

        _cache["metrics"][model_name] = metrics
        logger.info(f"Backtest {model_name}: MAPE={metrics.get('mape', '?'):.2f}%, "
                    f"dir_acc={metrics.get('directional_accuracy', '?')}%, "
                    f"coverage={metrics.get('coverage_rate', '?')}%")

    ensemble_backtest_metrics = {}
    if test_len > 0 and backtest_outputs:
        bt_ens_p50 = [0.0] * test_len
        bt_ens_p10 = [0.0] * test_len
        bt_ens_p90 = [0.0] * test_len
        total_bt_weight = 0.0
        for model_name, bt_pred in backtest_outputs.items():
            weight = model_weights.get(model_name, 0.0)
            p50 = bt_pred.get("p50", [])
            p10 = bt_pred.get("p10", p50)
            p90 = bt_pred.get("p90", p50)
            if weight > 0 and len(p50) >= test_len and len(p10) >= test_len and len(p90) >= test_len:
                for i in range(test_len):
                    bt_ens_p50[i] += weight * float(p50[i])
                    bt_ens_p10[i] += weight * float(p10[i])
                    bt_ens_p90[i] += weight * float(p90[i])
                total_bt_weight += weight
        if total_bt_weight > 0:
            bt_ens_p50 = [v / total_bt_weight for v in bt_ens_p50]
            bt_ens_p10 = [v / total_bt_weight for v in bt_ens_p10]
            bt_ens_p90 = [v / total_bt_weight for v in bt_ens_p90]
            ensemble_backtest_metrics = model_evaluator.evaluate(actual_test, bt_ens_p50, "ensemble")
            ensemble_backtest_metrics["directional_accuracy"] = model_evaluator.directional_accuracy(
                actual_test,
                bt_ens_p50,
                baseline_actual=baseline_price,
                baseline_predicted=baseline_price,
            )
            ensemble_backtest_metrics["coverage_rate"] = model_evaluator.coverage_rate(actual_test, bt_ens_p10, bt_ens_p90)

    segment_metrics = evaluate_forecast_segments(
        actual_test,
        backtest_outputs,
        baseline_price=baseline_price,
        segments=FORECAST_SEGMENTS,
    ) if test_len > 0 and backtest_outputs else {}
    _cache["segment_metrics"] = segment_metrics

    # === Metric-specialized ensemble ===
    # Price, direction and interval coverage come from the best model for each metric and forecast segment.
    valid_models = {
        n: m for n, m in _cache["metrics"].items()
        if (
            float(np.var(_cache["predictions"].get(n, {}).get("p50", [0]))) > 0.1
            and _cache["predictions"].get(n, {}).get("qa_passed", True)
        )
    }
    if not valid_models:
        logger.warning("No QA-passed models available for ensemble selection; using safe baseline")
        safe_pred = _safe_prediction_baseline(historical, settings.prediction_horizon)
        safe_valid, safe_summary, safe_details = qa_engine.validate_predictions(safe_pred, historical, "safe_baseline")
        safe_pred["qa_passed"] = safe_valid
        safe_pred["qa_summary"] = safe_summary
        safe_pred["qa_checks"] = safe_details.get("layer1_checks", [])
        safe_pred["excluded_from_ensemble"] = False
        _cache["predictions"]["safe_baseline"] = safe_pred
        _cache["metrics"]["safe_baseline"] = {
            "model": "safe_baseline",
            "mape": 999.0,
            "price_accuracy": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "directional_accuracy": 50.0,
            "coverage_rate": 100.0,
        }
        valid_models = {"safe_baseline": _cache["metrics"]["safe_baseline"]}

    excluded_models = {
        name: {
            "qa_summary": pred.get("qa_summary", ""),
            "failed_checks": _failed_qa_checks(pred.get("qa_checks", [])),
            "auto_repair": _cache.get("auto_repairs", {}).get(name),
        }
        for name, pred in _cache["predictions"].items()
        if name not in valid_models and name != "ensemble"
    }
    for name, pred in _cache["predictions"].items():
        if name != "ensemble":
            pred["excluded_from_ensemble"] = name not in valid_models
    _cache["excluded_models"] = excluded_models

    specialized_ensemble = build_metric_specialized_ensemble(
        _cache["predictions"],
        valid_models,
        current_price=float(historical[-1]),
        segment_metrics=segment_metrics,
        segments=FORECAST_SEGMENTS,
    )
    best_per_metric = specialized_ensemble["best_per_metric"]
    best_per_segment = specialized_ensemble.get("best_per_segment", {})
    best_mape_model = best_per_metric.get("price_accuracy") or list(_cache["predictions"].keys())[0]
    best_dir_model = best_per_metric.get("direction") or best_mape_model
    best_coverage_model = best_per_metric.get("coverage") or best_mape_model

    logger.info(f"Best per metric: price_accuracy={best_mape_model}, direction={best_dir_model}, coverage={best_coverage_model}")
    if best_per_segment:
        logger.info(f"Best per forecast segment: {best_per_segment}")

    ens_p50 = specialized_ensemble["p50"]
    ens_p10 = specialized_ensemble["p10"]
    ens_p90 = specialized_ensemble["p90"]

    calibrated = calibrate_forecast_volatility(ens_p50, ens_p10, ens_p90, historical)
    ens_p50 = calibrated["p50"]
    ens_p10 = calibrated["p10"]
    ens_p90 = calibrated["p90"]

    # Get best directional accuracy value
    best_dir_acc = specialized_ensemble.get(
        "direction_confidence",
        _cache["metrics"].get(best_dir_model, {}).get("directional_accuracy", 50.0),
    )

    # Step 3: News sentiment analysis with deterministic bounded adjustment
    current_price = float(historical[-1])
    try:
        news_sentiment = await news_sentiment_service.get_market_sentiment(
            llm_service=llm_service,
            current_price=current_price,
            limit=8,
        )
    except Exception as exc:
        logger.warning(f"News sentiment refresh failed, using neutral fallback: {exc}")
        news_sentiment = news_sentiment_service.analyze_rule_based([], current_price=current_price)
        news_sentiment["source"] = "neutral_fallback"
    _cache["news_sentiment"] = news_sentiment
    _cache["news_refresh_at"] = datetime.now().isoformat()

    # LLM output is advisory only; it must not overwrite p10/p50/p90.
    llm_optimization_applied = False
    _cache["llm_adjustment_proposal"] = None

    news_curve = news_sentiment_service.build_adjustment_curve(
        news_sentiment.get("price_adjustment_pct", 0.0),
        horizon=len(ens_p50),
    )
    news_sentiment_applied = any(abs(v) > 1e-9 for v in news_curve)
    if news_sentiment_applied:
        for i, adjustment_pct in enumerate(news_curve):
            ens_p50[i] = ens_p50[i] * (1 + adjustment_pct)
            ens_p10[i] = ens_p10[i] * (1 + adjustment_pct)
            ens_p90[i] = ens_p90[i] * (1 + adjustment_pct)
        news_sentiment["applied"] = True
        news_sentiment["adjustment_curve"] = news_curve[:7]
        _cache["news_sentiment"] = news_sentiment
        logger.info(f"News sentiment adjustment applied: {news_sentiment.get('price_adjustment_pct', 0) * 100:.2f}% short-end")

    ens_p50 = [round(v, 2) for v in ens_p50]
    ens_p10 = [round(v, 2) for v in ens_p10]
    ens_p90 = [round(v, 2) for v in ens_p90]

    # Keep the coverage model's interval shape, only enforce a minimum tier width.
    for i in range(len(ens_p50)):
        if i < 7:
            min_spread = ens_p50[i] * (0.008 + i * 0.002)
        elif i < 14:
            min_spread = ens_p50[i] * (0.02 + (i - 7) * 0.003)
        else:
            min_spread = ens_p50[i] * (0.04 + (i - 14) * 0.002)
        lower_spread = max(ens_p50[i] - ens_p10[i], min_spread)
        upper_spread = max(ens_p90[i] - ens_p50[i], min_spread)
        ens_p10[i] = round(ens_p50[i] - lower_spread, 2)
        ens_p90[i] = round(ens_p50[i] + upper_spread, 2)

    ensemble_pred = {
        "p50": ens_p50,
        "p10": ens_p10,
        "p90": ens_p90,
        "mean": ens_p50,
        "model": "综合预测",
        "price_model": best_mape_model,
        "direction_model": best_dir_model,
        "coverage_model": best_coverage_model,
        "best_per_segment": best_per_segment,
        "llm_optimization_applied": llm_optimization_applied,
        "llm_adjustment_proposal": None,
        "news_sentiment_applied": news_sentiment_applied,
        "news_adjustment_pct": news_sentiment.get("price_adjustment_pct", 0.0),
        "excluded_models": excluded_models,
    }
    ens_valid, ens_summary, ens_details = qa_engine.validate_predictions(ensemble_pred, historical, "ensemble")
    ensemble_pred["qa_passed"] = ens_valid
    ensemble_pred["qa_summary"] = ens_summary
    ensemble_pred["qa_checks"] = ens_details.get("layer1_checks", [])
    ensemble_pred["excluded_from_ensemble"] = False
    _cache["predictions"]["ensemble"] = ensemble_pred

    # Ensemble metrics describe the selected component for each decision dimension.
    ens_metrics = dict(valid_models.get(best_mape_model, ensemble_backtest_metrics))
    ens_metrics["model"] = "ensemble"
    ens_metrics["directional_accuracy"] = valid_models.get(best_dir_model, {}).get("directional_accuracy", 0.0)
    ens_metrics["coverage_rate"] = valid_models.get(best_coverage_model, {}).get("coverage_rate", 0.0)
    ens_metrics["price_model"] = best_mape_model
    ens_metrics["direction_model"] = best_dir_model
    ens_metrics["coverage_model"] = best_coverage_model
    _cache["metrics"]["ensemble"] = ens_metrics

    # Determine ensemble direction
    ens_direction = specialized_ensemble.get("direction", "震荡")
    dir_p50 = _cache["predictions"].get(best_dir_model, {}).get("p50", [])
    if len(dir_p50) >= 7 and historical[-1]:
        ens_change_pct = round((float(dir_p50[6]) - float(historical[-1])) / float(historical[-1]) * 100, 2)
    elif len(ens_p50) >= 7 and ens_p50[0]:
        ens_change_pct = round((ens_p50[6] - ens_p50[0]) / ens_p50[0] * 100, 2)
    else:
        ens_change_pct = 0
    _cache["ensemble_direction"] = ens_direction
    _cache["ensemble_change_pct"] = ens_change_pct
    _cache["best_per_metric"] = {
        "price_accuracy": best_mape_model,
        "direction": best_dir_model,
        "coverage": best_coverage_model,
    }
    _cache["best_per_segment"] = best_per_segment

    # Keep API startup responsive. Reports get a deterministic default payload
    # immediately, then the slower LLM versions refresh in the background.
    decision_models = {best_mape_model, best_dir_model, best_coverage_model, "ensemble"}
    qa_failures = {
        m: {"passed": p.get("qa_passed"), "summary": p.get("qa_summary")}
        for m, p in _cache["predictions"].items()
        if m in decision_models and not p.get("qa_passed", True)
    }
    historical_list = historical[-30:].tolist()
    _cache["report"] = _build_default_analysis_report(
        current_price=float(historical[-1]),
        predictions=ensemble_pred,
        model_metrics=ens_metrics,
        qa_summary=ens_summary,
        news_sentiment=news_sentiment,
    )
    _cache["risk_report"] = _build_default_risk_report(qa_failures, news_sentiment)
    _schedule_llm_report_refresh(
        current_price=float(historical[-1]),
        ensemble_pred=ensemble_pred,
        historical_prices=historical_list,
        ens_metrics=ens_metrics,
        ens_summary=ens_summary,
        news_sentiment=news_sentiment,
        qa_failures=qa_failures,
    )

    logger.info(f"Ensemble: metric-specialized, price={best_mape_model}, direction={best_dir_model} ({best_dir_acc:.0f}%), coverage={best_coverage_model}. LLM applied={llm_optimization_applied}, news applied={news_sentiment_applied}. Direction={ens_direction} ({ens_change_pct}%)")


def session_max_age(remember_me: bool) -> int:
    """Return cookie lifetime in seconds."""
    if remember_me:
        return settings.remember_me_days * 24 * 60 * 60
    return settings.session_idle_minutes * 60


def create_token(username: str, role: str, remember_me: bool = False) -> str:
    """Create a signed session token."""
    max_age = session_max_age(remember_me)
    payload = {
        "sub": username,
        "role": role,
        "remember_me": remember_me,
        "exp": datetime.utcnow() + timedelta(seconds=max_age),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def set_session_cookie(response: Response, token: str, remember_me: bool):
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=session_max_age(remember_me),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response):
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def verify_token(
    response: Response = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    session_token: str | None = Cookie(default=None, alias=settings.session_cookie_name),
) -> dict:
    """Verify bearer or HttpOnly cookie token and return user info."""
    raw_token = credentials.credentials if credentials else session_token
    if not raw_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        payload = jwt.decode(
            raw_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        remember_me = bool(payload.get("remember_me", False))
        if response is not None and session_token and not credentials:
            refreshed = create_token(payload["sub"], payload["role"], remember_me=remember_me)
            set_session_cookie(response, refreshed, remember_me)
        return {"username": payload["sub"], "role": payload["role"], "remember_me": remember_me}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _get_prediction_tier(day_index: int) -> dict:
    """Return confidence tier info for a given prediction day."""
    if day_index < 7:
        return {"tier": "precise", "tier_label": "精确预测"}
    elif day_index < 14:
        return {"tier": "standard", "tier_label": "标准预测"}
    else:
        return {"tier": "fuzzy", "tier_label": "模糊预测"}


def _build_prediction_points(
    pred: dict,
    data_end_date: date | None = None,
    display_start_date: date | None = None,
) -> list:
    """Build prediction point list with dates and tier labels."""
    p50 = pred.get("p50", [])
    p10 = pred.get("p10", p50)
    p90 = pred.get("p90", p50)
    display_start = display_start_date or date.today()
    if data_end_date is not None:
        forecast_targets = _build_visible_forecast_targets(data_end_date, display_start, len(p50))
    else:
        forecast_targets = list(enumerate(_build_forecast_dates(display_start, len(p50))))
    points = []
    for i, target_date in forecast_targets:
        tier = _get_prediction_tier(i)
        points.append({
            "target_date": str(target_date),
            "p10": round(float(p10[i] if i < len(p10) else p50[i]), 2),
            "p50": round(float(p50[i]), 2),
            "p90": round(float(p90[i] if i < len(p90) else p50[i]), 2),
            "source_index": i,
            **tier,
        })
    return points


def _get_prediction_date_context() -> tuple[date | None, date]:
    """Return latest real data date and the dashboard forecast display start."""
    price_df = _cache.get("price_data")
    display_start = date.today()
    if price_df is None or price_df.empty or "date" not in price_df.columns:
        return None, display_start
    latest_data_date = pd.to_datetime(price_df["date"]).dt.date.max()
    # Forecast arrays are generated from the first calendar day after the
    # latest real market observation. Official EIA/FRED feeds can lag by a few
    # days, so anchoring display to today would hide those leading forecast
    # offsets and make the chart appear disconnected.
    display_start = latest_data_date + timedelta(days=1)
    return latest_data_date, display_start


def _apply_historical_interval_floor(pred: dict, historical_prices: Any) -> dict:
    """Widen model intervals using only pre-forecast historical volatility."""
    p50 = np.asarray(pred.get("p50", []), dtype=float)
    if len(p50) == 0:
        return pred

    p10 = np.asarray(pred.get("p10", p50), dtype=float)
    p90 = np.asarray(pred.get("p90", p50), dtype=float)
    if len(p10) != len(p50):
        p10 = p50.copy()
    if len(p90) != len(p50):
        p90 = p50.copy()

    history = np.asarray(list(historical_prices), dtype=float)
    history = history[np.isfinite(history)]
    if len(history) < 8:
        updated = dict(pred)
        updated["p10"] = [round(float(v), 2) for v in np.minimum(p10, p50)]
        updated["p90"] = [round(float(v), 2) for v in np.maximum(p90, p50)]
        return updated

    recent = history[-min(len(history), 180):]
    changes = np.diff(recent)
    changes = changes[np.isfinite(changes)]
    if len(changes) >= 4:
        recent_move_floor = max(
            float(np.percentile(np.abs(changes), 80)),
            float(abs(p50[0]) * 0.008),
        )
    else:
        recent_move_floor = float(abs(p50[0]) * 0.01)

    for idx, center in enumerate(p50):
        horizon_scale = float(np.sqrt(idx + 1))
        pct_floor = abs(center) * min(0.08, 0.01 + idx * 0.0025)
        spread_floor = max(pct_floor, recent_move_floor * horizon_scale * 0.85)
        p10[idx] = min(p10[idx], center - spread_floor)
        p90[idx] = max(p90[idx], center + spread_floor)

    updated = dict(pred)
    updated["p10"] = [round(float(v), 2) for v in p10]
    updated["p90"] = [round(float(v), 2) for v in p90]
    updated["interval_calibrated"] = True
    return updated


def _evaluate_fixed_train_test_split() -> dict:
    """Evaluate models with separate validation and final test windows."""
    price_df = _cache.get("price_data")
    if price_df is None or price_df.empty:
        return {"status": "unavailable", "reason": "价格数据尚未加载"}

    price_df = price_df.sort_values("date").reset_index(drop=True)
    test_len = min(90, max(30, len(price_df) // 10))
    validation_len = min(90, max(30, (len(price_df) - test_len) // 10))
    embargo_len = 1 if len(price_df) > test_len + validation_len + 240 else 0

    train_core_end = len(price_df) - test_len - validation_len - embargo_len
    test_train_end = len(price_df) - test_len - embargo_len
    if train_core_end < 120 or test_train_end < 120:
        return {
            "status": "insufficient_data",
            "reason": "当前数据不足以进行训练/验证/测试三段评估",
            "train_rows": max(train_core_end, 0),
            "validation_rows": validation_len,
            "test_rows": test_len,
        }

    train_core = price_df.iloc[:train_core_end].copy()
    validation_price = price_df.iloc[train_core_end + embargo_len: train_core_end + embargo_len + validation_len].copy()
    test_train_price = price_df.iloc[:test_train_end].copy()
    test_price = price_df.iloc[test_train_end + embargo_len:].copy()

    def evaluate_window(train_price: pd.DataFrame, eval_price: pd.DataFrame, window_name: str) -> list[dict]:
        train_featured = feature_engineer.create_features(train_price, target_col="price")
        actual = eval_price["price"].astype(float).values
        horizon = len(actual)
        baseline_price = float(train_price["price"].iloc[-1])
        model_outputs = {}

        candidates = [
            ("naive", lambda: NaiveForecaster(prediction_horizon=horizon).predict(train_featured)),
            ("prophet", lambda: ProphetForecaster(prediction_horizon=horizon).train_and_predict(train_price)),
            ("xgboost", lambda: XGBoostForecaster(prediction_horizon=horizon).train_and_predict(train_featured)),
            ("tft", lambda: TFTModel(prediction_horizon=horizon)._fallback_predict(train_featured)),
        ]

        for model_name, runner in candidates:
            try:
                pred = runner()
                if len(pred.get("p50", [])) >= horizon:
                    model_outputs[model_name] = _apply_historical_interval_floor(pred, train_price["price"].values)
            except Exception as exc:
                logger.warning(f"Fixed split {window_name} evaluation failed for {model_name}: {exc}")

        rows = []
        for model_name, pred in model_outputs.items():
            p50 = pred["p50"][:horizon]
            p10 = pred["p10"][:horizon]
            p90 = pred["p90"][:horizon]
            metrics = model_evaluator.evaluate(actual, p50, model_name)
            metrics["directional_accuracy"] = model_evaluator.directional_accuracy(
                actual,
                p50,
                baseline_actual=baseline_price,
                baseline_predicted=baseline_price,
            )
            metrics["coverage_rate"] = model_evaluator.coverage_rate(actual, p10, p90)
            metrics["mean_interval_width_pct"] = round(
                float(np.mean((np.asarray(p90) - np.asarray(p10)) / np.maximum(np.asarray(p50), 1)) * 100),
                2,
            )
            metrics["interval_calibrated"] = bool(pred.get("interval_calibrated"))
            if model_name == "naive" and float(np.var(p50)) <= 0.1:
                metrics["directional_accuracy_applicable"] = False
                metrics["metric_notes"] = {
                    "directional_accuracy": "naive_flat_baseline",
                    "directional_accuracy_label": "平线基线不参与方向评分",
                }
            rows.append({"model_name": model_name, **metrics})
        return rows

    validation_models = evaluate_window(train_core, validation_price, "validation")
    models = evaluate_window(test_train_price, test_price, "test")
    best_model = min(validation_models, key=lambda row: row.get("mape", 999999))["model_name"] if validation_models else "N/A"
    oracle_test_best_model = min(models, key=lambda row: row.get("mape", 999999))["model_name"] if models else "N/A"
    selected_test_metrics = next((row for row in models if row.get("model_name") == best_model), None)

    validation_by_model = {row["model_name"]: row for row in validation_models}
    generalization_gaps = []
    for row in models:
        val = validation_by_model.get(row["model_name"])
        if not val:
            continue
        gap = round(float(row.get("mape", 0.0)) - float(val.get("mape", 0.0)), 4)
        generalization_gaps.append({
            "model_name": row["model_name"],
            "validation_mape": val.get("mape"),
            "test_mape": row.get("mape"),
            "mape_gap": gap,
            "coverage_gap": round(float(row.get("coverage_rate", 0.0)) - float(val.get("coverage_rate", 0.0)), 2),
        })
    max_gap = max([row["mape_gap"] for row in generalization_gaps], default=0.0)
    overfit_status = "warning" if max_gap > 10.0 else "pass"
    overfit_guard = {
        "method": "time_ordered_train_validation_test_with_1_day_embargo" if embargo_len else "time_ordered_train_validation_test",
        "status": overfit_status,
        "status_label": "需关注泛化差距" if overfit_status == "warning" else "通过",
        "max_mape_gap": round(max_gap, 4),
        "gaps": generalization_gaps,
    }
    return {
        "status": "ready",
        "train_window": f"{pd.to_datetime(train_core['date']).dt.date.min()} 至 {pd.to_datetime(train_core['date']).dt.date.max()}",
        "validation_window": f"{pd.to_datetime(validation_price['date']).dt.date.min()} 至 {pd.to_datetime(validation_price['date']).dt.date.max()}",
        "test_window": f"{pd.to_datetime(test_price['date']).dt.date.min()} 至 {pd.to_datetime(test_price['date']).dt.date.max()}",
        "train_rows": len(train_core),
        "validation_rows": len(validation_price),
        "test_rows": len(test_price),
        "embargo_rows": embargo_len,
        "best_model": best_model,
        "model_selection_basis": "validation_mape",
        "selected_test_metrics": selected_test_metrics,
        "oracle_test_best_model": oracle_test_best_model,
        "validation_models": validation_models,
        "models": models,
        "overfit_guard": overfit_guard,
    }


def _select_report_inputs() -> tuple:
    """Use the same ensemble/news context for initial and manual report generation."""
    predictions = _cache.get("predictions", {})
    metrics = _cache.get("metrics", {})
    if predictions.get("ensemble"):
        return (
            predictions["ensemble"],
            metrics.get("ensemble", {}),
            _cache.get("news_sentiment", {}),
        )

    best_model = min(
        metrics.items(),
        key=lambda x: x[1].get("mape", 999),
        default=("tft", {}),
    )
    return (
        predictions.get(best_model[0], {}),
        best_model[1],
        _cache.get("news_sentiment", {}),
    )


def _coerce_cached_evidence_bundle(value: Any = None) -> ForecastEvidenceBundle | None:
    """Return the cached evidence bundle as a model when possible."""
    bundle = _cache.get("forecast_evidence_bundle") if value is None else value
    if isinstance(bundle, ForecastEvidenceBundle):
        return bundle
    if isinstance(bundle, dict):
        try:
            return ForecastEvidenceBundle(**bundle)
        except Exception as exc:
            logger.warning(f"Cached evidence bundle is invalid: {exc}")
    return None


def _evidence_bundle_summary(bundle: ForecastEvidenceBundle | None) -> dict:
    """Build a compact, frontend-safe evidence summary."""
    if bundle is None:
        return {
            "commodity": "diesel_0",
            "as_of_date": str(date.today()),
            "current_price": None,
            "prediction_horizon": 0,
            "model_evidence_count": 0,
            "qa_evidence_count": 0,
            "news_evidence_count": 0,
            "risk_flag_count": 0,
            "evidence_ids": [],
        }

    evidence_ids = sorted(collect_evidence_ids(bundle))
    return convert_numpy({
        "commodity": bundle.commodity,
        "as_of_date": bundle.as_of_date,
        "current_price": bundle.current_price,
        "prediction_horizon": bundle.prediction_horizon,
        "model_evidence_count": len(bundle.model_evidence),
        "qa_evidence_count": len(bundle.qa_summary),
        "news_evidence_count": len(bundle.news_evidence),
        "risk_flag_count": len(bundle.risk_flags),
        "has_fixed_split_metrics": bundle.fixed_split_metrics is not None,
        "has_data_quality": bundle.data_quality is not None,
        "has_ensemble_rationale": bundle.ensemble_rationale is not None,
        "evidence_ids": evidence_ids,
    })


def _default_structured_report_payload() -> dict:
    """Return a safe structured-report shape when generation has not run."""
    return {
        "summary": "",
        "trend_view": "",
        "procurement_advice": {},
        "risk_flags": [],
        "confidence": None,
        "assumptions": [],
        "cited_evidence_ids": [],
        "model_limitations": [],
        "adjustment_proposal": None,
    }


async def _refresh_structured_report(commodity: str = "diesel_0"):
    """Refresh immutable evidence and advisory structured report without changing forecasts."""
    try:
        evidence_bundle = build_forecast_evidence_bundle(_cache, commodity=commodity)
        _cache["forecast_evidence_bundle"] = evidence_bundle
    except Exception as exc:
        logger.warning(f"Forecast evidence bundle generation failed: {exc}")
        _cache["forecast_evidence_bundle"] = None
        _cache["structured_report"] = None
        _cache["llm_adjustment_proposal"] = None
        ensemble = _cache.get("predictions", {}).get("ensemble")
        if isinstance(ensemble, dict):
            ensemble["llm_adjustment_proposal"] = None
        return None

    try:
        structured_report = await llm_service.generate_structured_analysis_report(evidence_bundle)
        _cache["structured_report"] = structured_report
        proposal = getattr(structured_report, "adjustment_proposal", None)
        proposal_payload = convert_numpy(proposal) if proposal else None
        _cache["llm_adjustment_proposal"] = proposal_payload
        ensemble = _cache.get("predictions", {}).get("ensemble")
        if isinstance(ensemble, dict):
            ensemble["llm_adjustment_proposal"] = proposal_payload
        return structured_report
    except Exception as exc:
        logger.warning(f"Structured report generation failed: {exc}")
        _cache["structured_report"] = None
        _cache["llm_adjustment_proposal"] = None
        ensemble = _cache.get("predictions", {}).get("ensemble")
        if isinstance(ensemble, dict):
            ensemble["llm_adjustment_proposal"] = None
        return None


# === API Routes ===

# --- Auth ---
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest, response: Response):
    """Authenticate user and set a HttpOnly session cookie."""
    engine = get_engine("sqlite:///./data/commodity_prediction.db")
    session = get_session(engine)

    user = session.query(User).filter(User.username == req.username).first()
    session.close()

    if not user or not pwd_context.verify(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token(user.username, user.role, remember_me=req.remember_me)
    set_session_cookie(response, token, remember_me=req.remember_me)
    return LoginResponse(
        role=user.role,
        username=user.username,
        session_expires_in=session_max_age(req.remember_me),
        remember_me=req.remember_me,
    )


@app.get("/api/auth/session")
async def get_session_state(
    response: Response,
    session_token: str | None = Cookie(default=None, alias=settings.session_cookie_name),
):
    """Return current cookie session state without logging a 401 for anonymous users."""
    if not session_token:
        return {"authenticated": False}

    try:
        payload = jwt.decode(
            session_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        clear_session_cookie(response)
        return {"authenticated": False}

    remember_me = bool(payload.get("remember_me", False))
    refreshed = create_token(payload["sub"], payload["role"], remember_me=remember_me)
    set_session_cookie(response, refreshed, remember_me)
    return {
        "authenticated": True,
        "username": payload["sub"],
        "role": payload["role"],
        "remember_me": remember_me,
    }


@app.get("/api/auth/me", response_model=UserInfo)
async def get_current_user(user=Depends(verify_token)):
    """Get current user information."""
    return UserInfo(username=user["username"], role=user["role"], remember_me=user.get("remember_me", False))


@app.post("/api/auth/logout")
async def logout(response: Response):
    """Clear the HttpOnly session cookie."""
    clear_session_cookie(response)
    return {"status": "logged_out"}


# --- Data ---
@app.get("/api/data/prices", response_model=PriceHistoryResponse)
async def get_price_history(days: int = 90, commodity: str = "diesel_0"):
    """Get historical price data."""
    df = _cache.get("price_data")
    if df is None:
        raise HTTPException(status_code=503, detail="Data not ready")

    df_recent = df.tail(days)

    data = [
        PricePoint(
            date=str(row["date"]),
            price=round(row["price"], 2),
            open_price=round(row.get("open", row["price"]), 2),
            high=round(row.get("high", row["price"]), 2),
            low=round(row.get("low", row["price"]), 2),
            volume=int(row.get("volume", 0)),
        )
        for _, row in df_recent.iterrows()
    ]

    return PriceHistoryResponse(
        commodity=commodity,
        data=data,
        total_records=len(data),
        date_range={
            "start": str(df_recent["date"].iloc[0]) if len(df_recent) > 0 else "",
            "end": str(df_recent["date"].iloc[-1]) if len(df_recent) > 0 else "",
        },
    )


@app.get("/api/data/latest", response_model=LatestPriceResponse)
async def get_latest_price(commodity: str = "diesel_0"):
    """Get the latest price."""
    price_df = _cache.get("price_data")
    if price_df is not None and not price_df.empty:
        latest = price_df.iloc[-1]
        prev = price_df.iloc[-2] if len(price_df) > 1 else latest
        change = float(latest["price"] - prev["price"])
        change_pct = (change / float(prev["price"]) * 100) if float(prev["price"]) else 0.0
        result = {
            "date": str(latest["date"]),
            "price": float(round(latest["price"], 2)),
            "change": round(change, 2),
            "change_pct": round(change_pct, 4),
        }
    else:
        result = await simulator.get_latest_price(commodity)
    return LatestPriceResponse(commodity=commodity, **result)


@app.get("/api/data/quality", response_model=DataQualityResponse)
async def get_data_quality():
    """Get data quality report."""
    report = _cache.get("data_quality")
    if not report:
        raise HTTPException(status_code=503, detail="Quality report not ready")
    return DataQualityResponse(**report)


# --- Predictions ---
@app.get("/api/predictions/latest")
async def get_latest_predictions(model: str = "tft"):
    """Get latest predictions from a specific model with 3-tier confidence."""
    pred = _cache["predictions"].get(model)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    latest_data_date, display_start_date = _get_prediction_date_context()
    return {
        "commodity": "diesel_0",
        "prediction_date": str(date.today()),
        "model_name": model,
        "predictions": _build_prediction_points(pred, latest_data_date, display_start_date),
        "is_qa_passed": pred.get("qa_passed", True),
        "qa_auto_repaired": pred.get("qa_auto_repaired", False),
        "excluded_from_ensemble": pred.get("excluded_from_ensemble", False),
        "qa_notes": pred.get("qa_summary", ""),
    }


@app.get("/api/predictions/all-models")
async def get_all_model_predictions():
    """Get predictions from all models for comparison."""
    result = {}
    latest_data_date, display_start_date = _get_prediction_date_context()
    for model_name, pred in _cache["predictions"].items():
        result[model_name] = _build_prediction_points(pred, latest_data_date, display_start_date)

    eligible_metrics = {
        name: metric
        for name, metric in _cache["metrics"].items()
        if (
            name == "ensemble"
            or (
                _cache["predictions"].get(name, {}).get("qa_passed", True)
                and not _cache["predictions"].get(name, {}).get("excluded_from_ensemble", False)
            )
        )
    } or _cache["metrics"]
    best = min(
        eligible_metrics.items(),
        key=lambda x: x[1].get("mape", 999),
        default=("tft", {})
    )

    return {
        "commodity": "diesel_0",
        "prediction_date": str(date.today()),
        "models": result,
        "recommended_model": best[0],
    }


# --- Model Metrics ---
@app.get("/api/metrics/comparison")
async def get_model_comparison():
    """Get performance comparison across all models."""
    metrics = _cache.get("metrics", {})

    if not metrics:
        return {"evaluation_date": str(date.today()), "models": [], "best_model": "N/A"}

    models = [
        {
            "model_name": name,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "model"},
            "qa_passed": _cache["predictions"].get(name, {}).get("qa_passed", True),
            "qa_auto_repaired": _cache["predictions"].get(name, {}).get("qa_auto_repaired", False),
            "excluded_from_ensemble": _cache["predictions"].get(name, {}).get("excluded_from_ensemble", False),
        }
        for name, m in metrics.items()
    ]

    eligible_metrics = {
        name: metric
        for name, metric in metrics.items()
        if (
            name == "ensemble"
            or (
                _cache["predictions"].get(name, {}).get("qa_passed", True)
                and not _cache["predictions"].get(name, {}).get("excluded_from_ensemble", False)
            )
        )
    } or metrics
    best = min(eligible_metrics.items(), key=lambda x: x[1].get("mape", 999))

    return {
        "evaluation_date": str(date.today()),
        "models": models,
        "best_model": best[0],
    }


# --- Analysis Reports ---
@app.get("/api/reports/latest")
async def get_latest_report():
    """Get the latest LLM analysis report."""
    report = _cache.get("report")
    if not report:
        return {
            "report_date": str(date.today()),
            "commodity": "diesel_0",
            "summary": "报告生成中...",
            "trend_analysis": "",
            "risk_factors": [],
            "procurement_advice": {},
            "data_quality_notes": "",
        }

    return {
        "report_date": str(date.today()),
        "commodity": "diesel_0",
        **report,
    }


@app.get("/api/reports/structured")
async def get_structured_report():
    """Get the schema-validated report and evidence references."""
    bundle = _coerce_cached_evidence_bundle()
    bundle_summary = _evidence_bundle_summary(bundle)
    evidence_ids = bundle_summary.get("evidence_ids", [])
    structured_report = _cache.get("structured_report")
    structured_payload = convert_numpy(structured_report) if structured_report else _default_structured_report_payload()
    adjustment_proposal = _cache.get("llm_adjustment_proposal")
    if adjustment_proposal is None and structured_report is not None:
        proposal = getattr(structured_report, "adjustment_proposal", None)
        adjustment_proposal = convert_numpy(proposal) if proposal else None

    return convert_numpy({
        "report_date": str(date.today()),
        "commodity": bundle.commodity if bundle else "diesel_0",
        "structured_report": structured_payload,
        "evidence_bundle_summary": bundle_summary,
        "evidence_ids": evidence_ids,
        "adjustment_proposal": adjustment_proposal,
    })


@app.get("/api/reports/risk")
async def get_risk_report():
    """Get the 3-dimensional risk research report."""
    report = _cache.get("risk_report")
    if not report:
        return {
            "report_date": str(date.today()),
            "dimension_1_market": [],
            "dimension_2_model": [],
            "dimension_3_policy": [],
        }
    return report


@app.post("/api/reports/regenerate")
async def regenerate_report(user=Depends(verify_token)):
    """Regenerate the analysis report."""
    if user["role"] not in ["admin", "executive"]:
        raise HTTPException(status_code=403, detail="权限不足")

    historical = _cache["price_data"]["price"].values if _cache.get("price_data") is not None else []
    historical_list = historical.tolist() if hasattr(historical, "tolist") else list(historical)
    report_pred, report_metrics, news_sentiment = _select_report_inputs()

    structured_report = await _refresh_structured_report()
    _cache["report"] = await llm_service.generate_analysis_report(
        current_price=float(historical[-1]) if len(historical) > 0 else 7800,
        predictions=report_pred,
        historical_prices=historical_list[-30:] if len(historical_list) >= 30 else historical_list,
        model_metrics=report_metrics,
        news_sentiment=news_sentiment,
    )

    return convert_numpy({
        "status": "regenerated",
        "report": _cache["report"],
        "structured_report": structured_report,
        "adjustment_proposal": _cache.get("llm_adjustment_proposal"),
    })


# --- Backtest ---
@app.get("/api/backtest/results")
async def get_backtest_results():
    """Get walk-forward backtest results (90-day window, 7-day step)."""
    df = _cache.get("featured_data")
    price_df = _cache.get("price_data")
    if df is None or price_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")

    prices = price_df["price"].values
    dates_arr = price_df["date"].values
    window_size = 90
    step_size = 7
    results = []
    procurement_periods = []

    def _pad_forecast(values, fallback_price: float, length: int) -> list:
        forecast = []
        for value in (values or [])[:length]:
            try:
                forecast.append(float(value))
            except (TypeError, ValueError):
                continue
        if not forecast:
            forecast = [float(fallback_price)]
        if len(forecast) < length:
            forecast.extend([forecast[-1]] * (length - len(forecast)))
        return forecast[:length]

    # Walk backward through the data, generating up to 10 backtest periods
    max_periods = 10
    for period_idx in range(max_periods):
        test_end = len(prices) - period_idx * step_size
        train_end = test_end - step_size
        train_start = train_end - window_size

        if train_start < 0 or train_end <= 0 or test_end <= train_end:
            break

        actual = prices[train_end:test_end].tolist()
        actual_dates = [pd.to_datetime(d).to_pydatetime() for d in dates_arr[train_end:test_end]]
        test_dates = [d.date().isoformat() for d in actual_dates]
        decision_time = pd.to_datetime(dates_arr[train_end - 1]).to_pydatetime()
        current_period_price = float(prices[train_end - 1])
        pred_p10 = []
        pred_p90 = []

        # Use XGBoost for backtest (fastest to retrain)
        try:
            train_df = df.iloc[:train_end].copy()
            if len(train_df) >= step_size * 3:
                xgb = XGBoostForecaster(prediction_horizon=step_size)
                pred_result = xgb.train_and_predict(train_df)
                predicted = _pad_forecast(pred_result.get("p50", []), current_period_price, len(actual))
                pred_p10 = _pad_forecast(pred_result.get("p10", predicted), current_period_price, len(actual))
                pred_p90 = _pad_forecast(pred_result.get("p90", predicted), current_period_price, len(actual))
            else:
                # Naive fallback: use last known price
                last_known = current_period_price if train_end > 0 else float(actual[0])
                predicted = [round(last_known, 2)] * len(actual)
                pred_p10 = [round(last_known * 0.99, 2)] * len(actual)
                pred_p90 = [round(last_known * 1.01, 2)] * len(actual)
        except Exception:
            last_known = current_period_price if train_end > 0 else float(actual[0])
            predicted = [round(last_known, 2)] * len(actual)
            pred_p10 = [round(last_known * 0.99, 2)] * len(actual)
            pred_p90 = [round(last_known * 1.01, 2)] * len(actual)

        rounded_actual = [round(float(v), 2) for v in actual]
        rounded_predicted = [round(float(v), 2) for v in predicted]
        rounded_p10 = [round(float(v), 2) for v in pred_p10]
        rounded_p90 = [round(float(v), 2) for v in pred_p90]

        results.append({
            "period_start": test_dates[0] if test_dates else "",
            "period_end": test_dates[-1] if test_dates else "",
            "dates": test_dates,
            "actual": rounded_actual,
            "predicted": rounded_predicted,
        })

        if rounded_actual and rounded_predicted and actual_dates:
            horizon_days = max(1, (actual_dates[-1].date() - decision_time.date()).days)
            procurement_periods.append({
                "instrument": "diesel_0",
                "decision_time": decision_time,
                "current_price": current_period_price,
                "predicted_prices": rounded_predicted,
                "p50": rounded_predicted[-1],
                "p10": rounded_p10[-1] if rounded_p10 else rounded_predicted[-1],
                "p90": rounded_p90[-1] if rounded_p90 else rounded_predicted[-1],
                "actual_prices": rounded_actual,
                "dates": actual_dates,
                "forecast_horizon_days": horizon_days,
                "quantity": 1.0,
                "metadata": {
                    "source": "api_backtest_results",
                    "period_start": test_dates[0] if test_dates else "",
                    "period_end": test_dates[-1] if test_dates else "",
                    "uses_future_prices_for_signal": False,
                },
            })

    procurement_result = run_procurement_backtest(
        list(reversed(procurement_periods)),
        instrument="diesel_0",
        quantity=1.0,
    )
    procurement_backtest = {
        "metrics": procurement_result.metrics,
        "procurement_savings": procurement_result.procurement_savings,
        "period_results": procurement_result.period_results,
        "signals": procurement_result.signals,
        "orders": procurement_result.orders,
        "fills": procurement_result.fills,
        "equity_curve": procurement_result.equity_curve,
    }

    return convert_numpy({
        "backtest_results": results,
        "procurement_backtest": procurement_backtest,
    })


@app.get("/api/backtest/fixed-split")
async def get_fixed_split_backtest():
    """Get 2024-2025 train / 2026 Jan-Apr validation results."""
    if not _cache.get("fixed_split_evaluation"):
        _cache["fixed_split_evaluation"] = _evaluate_fixed_train_test_split()
    return convert_numpy(_cache["fixed_split_evaluation"])


# --- Dashboard ---
@app.get("/api/dashboard/summary")
async def get_dashboard_summary(refresh: bool = False):
    """Get complete dashboard data in one call."""
    await _refresh_data_if_requested(refresh)
    price_df = _cache.get("price_data")

    if price_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")

    latest = price_df.iloc[-1]
    prev = price_df.iloc[-2] if len(price_df) > 1 else latest
    change = latest["price"] - prev["price"]
    change_pct = (change / prev["price"]) * 100 if prev["price"] > 0 else 0

    # Price history is anchored to the latest available real market date. Some
    # official feeds publish with a lag; anchoring to today can otherwise make
    # history look empty even when the dataset is healthy.
    history_end_date = pd.to_datetime(price_df["date"]).dt.date.max()
    history_rows = [
        {
            "date": str(row["date"]),
            "price": row["price"],
            "high": row.get("high", row["price"]),
            "low": row.get("low", row["price"]),
        }
        for _, row in price_df[pd.to_datetime(price_df["date"]).dt.date <= history_end_date].tail(90).iterrows()
    ]
    history = _build_calendar_price_history(history_rows, end_date=history_end_date, max_days=90)

    latest_data_date, display_start_date = _get_prediction_date_context()

    # Predictions with tier labels
    all_predictions = {}
    for model_name, pred in _cache["predictions"].items():
        all_predictions[model_name] = {
            "predictions": _build_prediction_points(pred, latest_data_date, display_start_date),
            "qa_passed": pred.get("qa_passed", True),
            "qa_checks": pred.get("qa_checks", []),
            "qa_auto_repaired": pred.get("qa_auto_repaired", False),
            "excluded_from_ensemble": pred.get("excluded_from_ensemble", False),
            "qa_summary": pred.get("qa_summary", ""),
        }

    # Model metrics
    metrics = [
        {"model_name": name, **{k: round(v, 4) if isinstance(v, float) else v
         for k, v in m.items() if k != "model"},
         "qa_passed": _cache["predictions"].get(name, {}).get("qa_passed", True),
         "qa_auto_repaired": _cache["predictions"].get(name, {}).get("qa_auto_repaired", False),
         "excluded_from_ensemble": _cache["predictions"].get(name, {}).get("excluded_from_ensemble", False)}
        for name, m in _cache.get("metrics", {}).items()
    ]

    # Risk alerts
    risk_alerts = []
    for model_name, pred in _cache["predictions"].items():
        if not pred.get("qa_passed", True):
            risk_alerts.append({
                "type": "qa_failure",
                "severity": "high",
                "model": model_name,
                "message": f"模型 {model_name} 未通过QA校验: {pred.get('qa_summary', '')}",
            })

    # Model disagreement alert
    if _cache.get("model_disagreement"):
        risk_alerts.append({
            "type": "model_disagreement",
            "severity": "medium",
            "message": "市场分歧风险：多个预测模型对价格走势方向不一致，建议增加人工研判权重",
        })

    # Price volatility alert
    recent_vol = price_df["price"].tail(10).std()
    long_vol = price_df["price"].tail(90).std()
    if recent_vol > long_vol * 1.5:
        risk_alerts.append({
            "type": "volatility",
            "severity": "medium",
            "message": f"近期价格波动加剧（10日σ={recent_vol:.0f} vs 90日σ={long_vol:.0f}）",
        })

    # KPIs — use ENSEMBLE metrics for executive dashboard
    ens_m = _cache.get("metrics", {}).get("ensemble", {})
    ens_p50 = _cache["predictions"].get("ensemble", {}).get("p50", [float(latest["price"])])
    visible_ensemble_points = all_predictions.get("ensemble", {}).get("predictions", [])
    visible_ensemble_p50 = [point["p50"] for point in visible_ensemble_points] or ens_p50
    ens_p50_7d = visible_ensemble_p50[:7]
    ens_avg = np.mean(ens_p50_7d) if ens_p50_7d else float(latest["price"])
    ens_change = ens_avg - latest["price"]
    ens_change_pct = (ens_change / latest["price"]) * 100 if latest["price"] > 0 else 0
    ens_mape = ens_m.get("mape", 2.0)
    ens_price_accuracy = ens_m.get("price_accuracy", max(0.0, 100.0 - ens_mape))
    ens_dir_acc = ens_m.get("directional_accuracy", 50.0)
    ens_coverage = ens_m.get("coverage_rate", 80.0)

    kpis = [
        {
            "title": "当前价格",
            "value": f"{latest['price']:.0f}",
            "change": f"{'+' if change >= 0 else ''}{change:.0f}",
            "change_direction": "up" if change > 0 else ("down" if change < 0 else "stable"),
            "unit": "RMB/吨",
        },
        {
            "title": "7日综合预测均价",
            "value": f"{ens_avg:.0f}",
            "change": f"较当前{'↑' if ens_change > 0 else '↓'}{abs(ens_change):.0f}元 ({abs(ens_change_pct):.2f}%)",
            "change_direction": "up" if ens_change > 0 else ("down" if ens_change < 0 else "stable"),
            "unit": "RMB/吨",
        },
        {
            "title": "7日价格准确率",
            "value": f"{ens_price_accuracy:.2f}%",
            "change": f"回测偏差率 {ens_mape:.2f}%",
            "change_direction": "stable",
            "unit": "",
        },
        {
            "title": "涨跌方向准确率",
            "value": f"{ens_dir_acc:.1f}%",
            "change": "近30日综合模型统计",
            "change_direction": "stable",
            "unit": "",
        },
    ]

    result = {
        "current_price": {
            "commodity": "diesel_0",
            "date": str(latest["date"]),
            "price": float(round(latest["price"], 2)),
            "change": float(round(change, 2)),
            "change_pct": float(round(change_pct, 4)),
        },
        "price_history": history,
        "predictions": all_predictions,
        "model_metrics": metrics,
        "risk_alerts": risk_alerts,
        "kpis": kpis,
        "analysis_report": _cache.get("report", {}),
        "risk_report": _cache.get("risk_report", {}),
        "news_sentiment": _cache.get("news_sentiment", {}),
        "data_quality": _cache.get("data_quality", {}),
        "data_source": _cache.get("data_source", "simulator"),
        "today": str(date.today()),
        "price_signature": _cache.get("price_signature"),
        "last_refresh_at": _cache.get("last_refresh_at"),
        "last_refresh_checked_at": _cache.get("last_refresh_checked_at"),
        "news_refresh_at": _cache.get("news_refresh_at"),
        "llm_models": {
            "provider": settings.resolved_llm_provider,
            "model": getattr(llm_service, "model", settings.resolved_llm_model),
            "base_url": settings.resolved_llm_base_url,
            "available": llm_service.is_available(),
        },
        "auto_repairs": _cache.get("auto_repairs", {}),
        "excluded_models": _cache.get("excluded_models", {}),
        "ensemble_direction": _cache.get("ensemble_direction", "震荡"),
        "ensemble_change_pct": _cache.get("ensemble_change_pct", 0),
        "best_per_metric": _cache.get("best_per_metric", {}),
        "best_per_segment": _cache.get("best_per_segment", {}),
        "segment_metrics": _cache.get("segment_metrics", {}),
        "ensemble_coverage": ens_coverage,
        "fixed_split_evaluation": _cache.get("fixed_split_evaluation"),
    }
    return convert_numpy(result)


# --- AI Chat (Procurement Assistant) ---
@app.post("/api/chat")
async def chat_with_ai(request: dict):
    """AI chatbot for procurement staff to analyze data."""
    question = request.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="请输入问题")

    # Build context from cached data
    price_df = _cache.get("price_data")
    current = float(price_df.iloc[-1]["price"]) if price_df is not None else 0
    predictions = _cache.get("predictions", {})
    metrics = _cache.get("metrics", {})
    news_sentiment = _cache.get("news_sentiment", {})

    # Get best model predictions
    best_name = "tft"
    best_p50 = []
    for name, pred in predictions.items():
        p50 = pred.get("p50", [])
        if len(p50) > 1 and float(np.var(p50)) > 1:
            best_name = name
            best_p50 = p50
            break

    context = f"""你是一位专业的大宗商品采购分析助手。以下是当前系统数据：

当前柴油价格：{current:.0f} RMB/吨
最优预测模型：{best_name}
7天预测价格：{[round(v, 0) for v in best_p50[:7]]}
30天预测价格趋势：{'上涨' if len(best_p50) > 1 and best_p50[-1] > best_p50[0] else '下跌'}
模型MAPE：{metrics.get(best_name, {}).get('mape', 'N/A')}%
新闻情绪：{news_sentiment.get('summary', '暂无新闻情绪')}

请基于以上数据回答用户的采购相关问题。回答要专业、简洁、可执行，使用中文。"""

    prompt = f"{context}\n\n用户问题：{question}"
    system = "你是一位拥有30年经验的大宗商品采购顾问。回答简洁专业，直接给出可执行建议。"

    raw = llm_service._call_llm(system, prompt, temperature=0.3, max_tokens=800)
    if raw:
        return {"answer": raw, "model_used": best_name, "current_price": current}
    return {"answer": "采购问答暂时不可用，请稍后重试。", "model_used": best_name, "current_price": current}


# --- Health ---
@app.get("/api/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "data_loaded": _cache.get("price_data") is not None,
        "models_ready": len(_cache.get("predictions", {})) > 0,
        "llm_available": llm_service.is_available(),
        "llm_models": {
            "provider": settings.resolved_llm_provider,
            "model": getattr(llm_service, "model", settings.resolved_llm_model),
            "base_url": settings.resolved_llm_base_url,
            "available": llm_service.is_available(),
        },
        "data_source": _cache.get("data_source", "unknown"),
    }


# --- Serve Frontend ---
# Root path serves index.html, static files served from /frontend/
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_index():
    """Serve the frontend index.html at root."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files AFTER all API routes so /api/* takes priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
