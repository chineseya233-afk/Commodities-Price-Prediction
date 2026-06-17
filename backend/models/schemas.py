"""
API Schemas — Pydantic models for request/response validation (Backend Agent)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date


# === Auth ===
class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False

class LoginResponse(BaseModel):
    role: str
    username: str
    session_expires_in: int
    remember_me: bool = False

class UserInfo(BaseModel):
    username: str
    role: str
    full_name: Optional[str] = None
    remember_me: bool = False


# === Price Data ===
class PricePoint(BaseModel):
    date: str
    price: float
    open_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None

class PriceHistoryResponse(BaseModel):
    commodity: str
    data: List[PricePoint]
    total_records: int
    date_range: Dict[str, str]

class LatestPriceResponse(BaseModel):
    commodity: str
    date: str
    price: float
    change: float
    change_pct: float


# === Predictions ===
class PredictionPoint(BaseModel):
    target_date: str
    p10: float
    p50: float
    p90: float

class PredictionResponse(BaseModel):
    commodity: str
    prediction_date: str
    model_name: str
    predictions: List[PredictionPoint]
    is_qa_passed: bool = True
    qa_notes: Optional[str] = None

class MultiModelPredictionResponse(BaseModel):
    commodity: str
    prediction_date: str
    models: Dict[str, List[PredictionPoint]]
    recommended_model: str


# === Model Metrics ===
class ModelMetricsResponse(BaseModel):
    model_name: str
    mape: float
    rmse: float
    mae: float
    directional_accuracy: float
    coverage_rate: Optional[float] = None

class ModelComparisonResponse(BaseModel):
    evaluation_date: str
    models: List[ModelMetricsResponse]
    best_model: str


# === Analysis Reports ===
class AnalysisReportResponse(BaseModel):
    report_date: str
    commodity: str
    summary: str
    trend_analysis: str
    risk_factors: List[str]
    procurement_advice: Dict[str, Any]
    data_quality_notes: Optional[str] = None


# === LLM Evidence Bundles ===
class EvidenceItem(BaseModel):
    evidence_id: str
    source: str
    title: str
    value: Any
    timestamp: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelEvidence(BaseModel):
    evidence_id: str
    model_name: str
    prediction_summary: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    segment_metrics: Dict[str, Any] = Field(default_factory=dict)
    coverage: Dict[str, Any] = Field(default_factory=dict)
    explainability: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    segment_evidence_ids: Dict[str, str] = Field(default_factory=dict)


class NewsEvidence(BaseModel):
    evidence_id: str
    source: str
    title: str
    sentiment: Any = None
    impact: Optional[Any] = None
    impact_score: Optional[float] = None
    source_url: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastEvidenceBundle(BaseModel):
    commodity: str
    as_of_date: str
    current_price: float
    prediction_horizon: int
    model_evidence: List[ModelEvidence] = Field(default_factory=list)
    qa_summary: List[EvidenceItem] = Field(default_factory=list)
    fixed_split_metrics: Optional[EvidenceItem] = None
    news_evidence: List[NewsEvidence] = Field(default_factory=list)
    data_quality: Optional[EvidenceItem] = None
    ensemble_rationale: Optional[EvidenceItem] = None
    risk_flags: List[EvidenceItem] = Field(default_factory=list)


class AdjustmentProposal(BaseModel):
    recommendation: str
    suggested_bias_pct: Optional[float] = None
    rationale: Optional[str] = None
    cited_evidence_ids: List[str] = Field(default_factory=list)
    review_required: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class StructuredAnalysisReport(BaseModel):
    summary: str
    trend_view: str
    procurement_advice: Dict[str, Any]
    risk_flags: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    model_limitations: List[str] = Field(default_factory=list)
    adjustment_proposal: Optional[AdjustmentProposal] = None


# === Data Quality ===
class DataQualityResponse(BaseModel):
    total_records: int
    date_range: Dict[str, str]
    completeness: float
    target_stats: Dict[str, float]
    outlier_count: int
    outlier_pct: float


# === Dashboard ===
class DashboardSummary(BaseModel):
    current_price: LatestPriceResponse
    prediction_7d: PredictionResponse
    model_metrics: List[ModelMetricsResponse]
    risk_alerts: List[Dict[str, Any]]
    analysis_summary: Optional[str] = None

class KPICard(BaseModel):
    title: str
    value: str
    change: Optional[str] = None
    change_direction: Optional[str] = None  # up, down, stable
    unit: Optional[str] = None
