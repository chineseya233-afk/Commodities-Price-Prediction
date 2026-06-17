"""Build deterministic evidence bundles for structured LLM reports.

This module does not call an LLM and does not perform network I/O. It only
normalizes the cached forecast context into a contract that can be validated
before and after an LLM report is generated.
"""

from datetime import date, datetime
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from backend.models.schemas import (
    EvidenceItem,
    ForecastEvidenceBundle,
    ModelEvidence,
    NewsEvidence,
    StructuredAnalysisReport,
)


DEFAULT_COMMODITY = "\u67f4\u6cb9"


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_float_list(values: Iterable) -> List[float]:
    result: List[float] = []
    if values is None:
        return result
    for value in values:
        number = _to_float(value)
        if number is not None:
            result.append(round(number, 4))
    return result


def _to_confidence(value: Any) -> Optional[float]:
    confidence = _to_float(value)
    if confidence is None:
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(confidence, 1.0))


def _slug(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:48]


def _stable_evidence_id(kind: str, *parts: Any) -> str:
    payload = json.dumps(
        [kind, *[_jsonable(part) for part in parts]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    label = next((_slug(part) for part in parts if _slug(part)), "")
    return f"{kind}:{label}:{digest}" if label else f"{kind}:{digest}"


def _date_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    return text.split("T", 1)[0].split(" ", 1)[0]


def _latest_from_price_data(price_data: Any, field: str) -> Any:
    if price_data is None:
        return None

    columns = getattr(price_data, "columns", [])
    if field in columns and not getattr(price_data, "empty", True):
        try:
            return price_data[field].iloc[-1]
        except (AttributeError, IndexError, KeyError):
            return None

    if isinstance(price_data, Mapping):
        if field in price_data:
            return price_data[field]
        rows = price_data.get("data")
        if isinstance(rows, list) and rows:
            return _as_mapping(rows[-1]).get(field)

    if isinstance(price_data, list) and price_data:
        return _as_mapping(price_data[-1]).get(field)

    return None


def _extract_as_of_date(cache: Mapping[str, Any], as_of_date: Any) -> str:
    return (
        _date_string(as_of_date)
        or _date_string(cache.get("as_of_date"))
        or _date_string(cache.get("last_refresh_at"))
        or _date_string(_latest_from_price_data(cache.get("price_data"), "date"))
        or date.today().isoformat()
    )


def _extract_current_price(cache: Mapping[str, Any], selected_prediction: Mapping[str, Any]) -> float:
    direct = _to_float(cache.get("current_price"))
    if direct is not None:
        return direct

    latest_price = _as_mapping(cache.get("latest_price"))
    from_latest = _to_float(latest_price.get("price"))
    if from_latest is not None:
        return from_latest

    from_price_data = _to_float(_latest_from_price_data(cache.get("price_data"), "price"))
    if from_price_data is not None:
        return from_price_data

    p50 = _as_float_list(selected_prediction.get("p50", []))
    return float(p50[0]) if p50 else 0.0


def _select_prediction(cache: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    predictions = _as_mapping(cache.get("predictions"))
    if isinstance(predictions.get("ensemble"), Mapping):
        return "ensemble", _as_mapping(predictions["ensemble"])

    if not predictions:
        return "", {}

    metrics = _as_mapping(cache.get("metrics"))
    metric_candidates = [
        (name, metric)
        for name, metric in metrics.items()
        if name in predictions and isinstance(metric, Mapping)
    ]
    if metric_candidates:
        best_name = min(metric_candidates, key=lambda item: item[1].get("mape", 999999))[0]
        return str(best_name), _as_mapping(predictions.get(best_name))

    first_name = next(iter(predictions))
    return str(first_name), _as_mapping(predictions.get(first_name))


def _prediction_horizon(prediction: Mapping[str, Any]) -> int:
    return max(
        len(_as_float_list(prediction.get("p50", []))),
        len(_as_float_list(prediction.get("p10", []))),
        len(_as_float_list(prediction.get("p90", []))),
    )


def _prediction_summary(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    p50 = _as_float_list(prediction.get("p50", []))
    p10 = _as_float_list(prediction.get("p10", []))
    p90 = _as_float_list(prediction.get("p90", []))
    summary: Dict[str, Any] = {
        "horizon": len(p50),
        "p50_7d": p50[:7],
        "p10_7d": p10[:7],
        "p90_7d": p90[:7],
    }
    if p50:
        summary.update({
            "p50_first": p50[0],
            "p50_last": p50[-1],
            "p50_min": min(p50),
            "p50_max": max(p50),
        })
    for key in ("direction", "direction_confidence", "news_sentiment_applied", "llm_optimization_applied"):
        if key in prediction:
            summary[key] = _jsonable(prediction[key])
    return summary


def _model_segment_metrics(cache: Mapping[str, Any], model_name: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    segment_metrics = _as_mapping(cache.get("segment_metrics"))
    metrics_by_segment: Dict[str, Any] = {}
    ids_by_segment: Dict[str, str] = {}
    for segment_key, model_metrics in segment_metrics.items():
        model_metric = _as_mapping(model_metrics).get(model_name)
        if model_metric is None:
            continue
        segment_name = str(segment_key)
        metrics_by_segment[segment_name] = _jsonable(model_metric)
        ids_by_segment[segment_name] = _stable_evidence_id("segment", model_name, segment_name)
    return metrics_by_segment, ids_by_segment


def _coverage_summary(prediction: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    for key in ("coverage_rate", "mean_interval_width_pct"):
        if key in metrics:
            coverage[key] = _jsonable(metrics[key])
    p10 = _as_float_list(prediction.get("p10", []))
    p90 = _as_float_list(prediction.get("p90", []))
    if p10 and p90:
        coverage["interval_7d"] = [
            {"p10": low, "p90": high}
            for low, high in zip(p10[:7], p90[:7])
        ]
    return coverage


def _explainability_summary(cache: Mapping[str, Any], model_name: str, prediction: Mapping[str, Any]) -> Dict[str, Any]:
    explainability: Dict[str, Any] = {}
    for key in ("model", "feature_importance", "interpretation", "attention", "best_per_metric", "best_per_segment"):
        if key in prediction:
            explainability[key] = _jsonable(prediction[key])
    if model_name == "ensemble":
        for key in ("best_per_metric", "best_per_segment", "ensemble_direction", "ensemble_change_pct"):
            if key in cache:
                explainability[key] = _jsonable(cache[key])
    return explainability


def _build_model_evidence(cache: Mapping[str, Any]) -> List[ModelEvidence]:
    predictions = _as_mapping(cache.get("predictions"))
    metrics = _as_mapping(cache.get("metrics"))
    model_names = sorted({str(name) for name in predictions.keys()} | {str(name) for name in metrics.keys()})
    evidence: List[ModelEvidence] = []

    for model_name in model_names:
        prediction = _as_mapping(predictions.get(model_name))
        metric = _as_mapping(metrics.get(model_name))
        segment_metric, segment_ids = _model_segment_metrics(cache, model_name)
        notes = prediction.get("qa_summary") or metric.get("notes")
        evidence.append(
            ModelEvidence(
                evidence_id=_stable_evidence_id("model", model_name),
                model_name=model_name,
                prediction_summary=_prediction_summary(prediction),
                metrics=_jsonable(metric),
                segment_metrics=segment_metric,
                coverage=_coverage_summary(prediction, metric),
                explainability=_explainability_summary(cache, model_name, prediction),
                notes=str(notes) if notes else None,
                segment_evidence_ids=segment_ids,
            )
        )

    return evidence


def _build_qa_evidence(cache: Mapping[str, Any], as_of_date: str) -> List[EvidenceItem]:
    predictions = _as_mapping(cache.get("predictions"))
    evidence: List[EvidenceItem] = []

    for model_name in sorted(str(name) for name in predictions.keys()):
        prediction = _as_mapping(predictions.get(model_name))
        if not any(key in prediction for key in ("qa_passed", "qa_summary", "qa_checks")):
            continue
        passed = prediction.get("qa_passed", True)
        evidence.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("qa", model_name),
                source="qa_engine",
                title=f"{model_name} QA summary",
                value={
                    "model_name": model_name,
                    "passed": bool(passed),
                    "summary": prediction.get("qa_summary", ""),
                    "checks": _jsonable(prediction.get("qa_checks", [])),
                },
                timestamp=as_of_date,
                confidence=1.0 if passed else 0.4,
                metadata={"model_name": model_name},
            )
        )

    if cache.get("qa_summary") is not None:
        evidence.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("qa", "aggregate"),
                source="qa_engine",
                title="Aggregate QA summary",
                value=_jsonable(cache.get("qa_summary")),
                timestamp=as_of_date,
                confidence=1.0,
            )
        )

    return evidence


def _build_fixed_split_evidence(cache: Mapping[str, Any], as_of_date: str) -> Optional[EvidenceItem]:
    fixed_split = cache.get("fixed_split_evaluation")
    if fixed_split is None:
        return None
    return EvidenceItem(
        evidence_id=_stable_evidence_id("fixed-split", "evaluation"),
        source="model_evaluator",
        title="Fixed train/test split evaluation",
        value=_jsonable(fixed_split),
        timestamp=as_of_date,
        confidence=1.0 if _as_mapping(fixed_split).get("status") == "ready" else None,
    )


def _impact_score(news: Mapping[str, Any], item: Mapping[str, Any]) -> Optional[float]:
    for key in ("impact_score", "score"):
        score = _to_float(item.get(key))
        if score is not None:
            return score
    adjustment = _to_float(news.get("price_adjustment_pct"))
    return abs(adjustment) if adjustment is not None else None


def _build_news_evidence(cache: Mapping[str, Any], as_of_date: str) -> List[NewsEvidence]:
    news = _as_mapping(cache.get("news_sentiment"))
    if not news:
        return []

    raw_items = news.get("news_items")
    items = raw_items if isinstance(raw_items, list) else []
    if not items and any(key in news for key in ("summary", "sentiment_score", "price_adjustment_pct")):
        items = [{
            "title": news.get("summary") or "News sentiment summary",
            "sentiment": news.get("sentiment_score"),
            "impact_score": _impact_score(news, {}),
            "source": "news_sentiment_service",
        }]

    evidence: List[NewsEvidence] = []
    for index, raw_item in enumerate(items):
        item = _as_mapping(raw_item)
        title = str(item.get("title") or news.get("summary") or "News sentiment item")
        source_url = item.get("source_url") or item.get("url")
        source = str(item.get("source") or "news_sentiment_service")
        evidence.append(
            NewsEvidence(
                evidence_id=_stable_evidence_id("news", source_url or title, index),
                source=source,
                title=title,
                sentiment=item.get("sentiment", item.get("direction", news.get("sentiment_score"))),
                impact=item.get("impact", item.get("summary", news.get("summary"))),
                impact_score=_impact_score(news, item),
                source_url=str(source_url) if source_url else None,
                timestamp=_date_string(item.get("timestamp") or item.get("published_at") or item.get("date")) or as_of_date,
                metadata={
                    "aggregate_summary": news.get("summary"),
                    "price_adjustment_pct": _jsonable(news.get("price_adjustment_pct")),
                    "raw_item": _jsonable(item),
                },
            )
        )

    return evidence


def _build_data_quality_evidence(cache: Mapping[str, Any], as_of_date: str) -> Optional[EvidenceItem]:
    data_quality = _as_mapping(cache.get("data_quality"))
    if not data_quality:
        return None

    completeness = _to_float(data_quality.get("completeness"))
    confidence = None
    if completeness is not None:
        confidence = max(0.0, min(completeness / 100.0, 1.0))

    return EvidenceItem(
        evidence_id=_stable_evidence_id("data-quality", "latest"),
        source="data_preprocessor",
        title="Data quality report",
        value=_jsonable(data_quality),
        timestamp=as_of_date,
        confidence=confidence,
    )


def _build_ensemble_rationale(cache: Mapping[str, Any], selected_model: str, as_of_date: str) -> Optional[EvidenceItem]:
    predictions = _as_mapping(cache.get("predictions"))
    ensemble = _as_mapping(predictions.get("ensemble"))
    value = {
        "selected_model": selected_model,
        "best_per_metric": _jsonable(cache.get("best_per_metric") or ensemble.get("best_per_metric")),
        "best_per_segment": _jsonable(cache.get("best_per_segment") or ensemble.get("best_per_segment")),
        "model_disagreement": bool(cache.get("model_disagreement", False)),
        "ensemble_prediction_flags": {
            "news_sentiment_applied": bool(ensemble.get("news_sentiment_applied", False)),
            "llm_optimization_applied": bool(ensemble.get("llm_optimization_applied", False)),
            "price_model": ensemble.get("price_model"),
            "direction_model": ensemble.get("direction_model"),
            "coverage_model": ensemble.get("coverage_model"),
        },
    }
    if not ensemble and not value["best_per_metric"] and not value["best_per_segment"]:
        return None

    return EvidenceItem(
        evidence_id=_stable_evidence_id("ensemble", "rationale"),
        source="ensemble_builder",
        title="Metric-specialized ensemble rationale",
        value=value,
        timestamp=as_of_date,
        confidence=1.0,
    )


def _explicit_risk_flags(cache: Mapping[str, Any], as_of_date: str) -> List[EvidenceItem]:
    raw_flags = cache.get("risk_flags")
    if not isinstance(raw_flags, list):
        return []

    evidence: List[EvidenceItem] = []
    for index, raw_flag in enumerate(raw_flags):
        flag = _as_mapping(raw_flag)
        title = str(flag.get("title") or flag.get("name") or raw_flag)
        evidence.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("risk", "explicit", index, title),
                source=str(flag.get("source") or "cache"),
                title=title,
                value=_jsonable(raw_flag),
                timestamp=_date_string(flag.get("timestamp")) or as_of_date,
                confidence=_to_confidence(flag.get("confidence")),
                metadata={"index": index},
            )
        )
    return evidence


def _derived_risk_flags(cache: Mapping[str, Any], as_of_date: str) -> List[EvidenceItem]:
    predictions = _as_mapping(cache.get("predictions"))
    news = _as_mapping(cache.get("news_sentiment"))
    data_quality = _as_mapping(cache.get("data_quality"))
    flags: List[EvidenceItem] = []

    for model_name in sorted(str(name) for name in predictions.keys()):
        prediction = _as_mapping(predictions.get(model_name))
        if prediction.get("qa_passed") is False:
            flags.append(
                EvidenceItem(
                    evidence_id=_stable_evidence_id("risk", "qa-failed", model_name),
                    source="qa_engine",
                    title=f"{model_name} QA failure",
                    value={"model_name": model_name, "summary": prediction.get("qa_summary", "")},
                    timestamp=as_of_date,
                    confidence=0.8,
                )
            )

    if cache.get("model_disagreement"):
        flags.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("risk", "model-disagreement"),
                source="ensemble_builder",
                title="Model direction disagreement",
                value=True,
                timestamp=as_of_date,
                confidence=0.7,
            )
        )

    adjustment = _to_float(news.get("price_adjustment_pct"))
    if adjustment is not None and abs(adjustment) > 0:
        flags.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("risk", "news-adjustment"),
                source="news_sentiment_service",
                title="News sentiment price adjustment",
                value={"price_adjustment_pct": adjustment, "summary": news.get("summary")},
                timestamp=as_of_date,
                confidence=0.6,
            )
        )

    completeness = _to_float(data_quality.get("completeness"))
    outlier_pct = _to_float(data_quality.get("outlier_pct"))
    if (completeness is not None and completeness < 95.0) or (outlier_pct is not None and outlier_pct > 5.0):
        flags.append(
            EvidenceItem(
                evidence_id=_stable_evidence_id("risk", "data-quality"),
                source="data_preprocessor",
                title="Data quality requires review",
                value={"completeness": completeness, "outlier_pct": outlier_pct},
                timestamp=as_of_date,
                confidence=0.7,
            )
        )

    return flags


def build_forecast_evidence_bundle(
    cache: Mapping[str, Any],
    commodity: str = DEFAULT_COMMODITY,
    as_of_date: Any = None,
) -> ForecastEvidenceBundle:
    """Assemble cached forecast context into a deterministic evidence bundle."""
    cache = _as_mapping(cache)
    selected_model, selected_prediction = _select_prediction(cache)
    resolved_as_of_date = _extract_as_of_date(cache, as_of_date)
    current_price = _extract_current_price(cache, selected_prediction)

    return ForecastEvidenceBundle(
        commodity=commodity,
        as_of_date=resolved_as_of_date,
        current_price=current_price,
        prediction_horizon=_prediction_horizon(selected_prediction),
        model_evidence=_build_model_evidence(cache),
        qa_summary=_build_qa_evidence(cache, resolved_as_of_date),
        fixed_split_metrics=_build_fixed_split_evidence(cache, resolved_as_of_date),
        news_evidence=_build_news_evidence(cache, resolved_as_of_date),
        data_quality=_build_data_quality_evidence(cache, resolved_as_of_date),
        ensemble_rationale=_build_ensemble_rationale(cache, selected_model, resolved_as_of_date),
        risk_flags=[
            *_explicit_risk_flags(cache, resolved_as_of_date),
            *_derived_risk_flags(cache, resolved_as_of_date),
        ],
    )


def collect_evidence_ids(bundle: ForecastEvidenceBundle) -> Set[str]:
    """Return every citeable evidence id in a bundle."""
    evidence_ids: Set[str] = set()

    for item in bundle.model_evidence:
        evidence_ids.add(item.evidence_id)
        evidence_ids.update(item.segment_evidence_ids.values())
    for item in bundle.qa_summary:
        evidence_ids.add(item.evidence_id)
    if bundle.fixed_split_metrics:
        evidence_ids.add(bundle.fixed_split_metrics.evidence_id)
    for item in bundle.news_evidence:
        evidence_ids.add(item.evidence_id)
    if bundle.data_quality:
        evidence_ids.add(bundle.data_quality.evidence_id)
    if bundle.ensemble_rationale:
        evidence_ids.add(bundle.ensemble_rationale.evidence_id)
    for item in bundle.risk_flags:
        evidence_ids.add(item.evidence_id)

    return evidence_ids


def missing_report_citations(
    report: StructuredAnalysisReport,
    bundle: ForecastEvidenceBundle,
) -> List[str]:
    """List report citations that are not present in the evidence bundle."""
    evidence_ids = collect_evidence_ids(bundle)
    cited_evidence_ids = list(report.cited_evidence_ids)
    if report.adjustment_proposal:
        cited_evidence_ids.extend(report.adjustment_proposal.cited_evidence_ids)
    return [evidence_id for evidence_id in cited_evidence_ids if evidence_id not in evidence_ids]


def validate_report_citations(
    report: StructuredAnalysisReport,
    bundle: ForecastEvidenceBundle,
) -> bool:
    """Return True when every report citation exists in the evidence bundle."""
    return not missing_report_citations(report, bundle)


class ReportContextService:
    """Service wrapper for callers that prefer dependency-injected classes."""

    def build_forecast_evidence_bundle(
        self,
        cache: Mapping[str, Any],
        commodity: str = DEFAULT_COMMODITY,
        as_of_date: Any = None,
    ) -> ForecastEvidenceBundle:
        return build_forecast_evidence_bundle(cache, commodity=commodity, as_of_date=as_of_date)
