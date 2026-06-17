"""Utilities for metric-specialized forecast composition."""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

DEFAULT_FORECAST_SEGMENTS = [
    {"key": "1-7天", "start": 0, "end": 7},
    {"key": "8-14天", "start": 7, "end": 14},
    {"key": "15-30天", "start": 14, "end": 30},
]


def _as_float_list(values: Iterable, horizon: int = None) -> List[float]:
    result = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result[:horizon] if horizon is not None else result


def _candidate_models(predictions: Dict, metrics: Dict) -> Dict:
    candidates = {}
    for name, metric in (metrics or {}).items():
        if name == "ensemble":
            continue
        pred = (predictions or {}).get(name, {})
        p50 = _as_float_list(pred.get("p50", []))
        if p50:
            candidates[name] = metric or {}
    return candidates


def _finite_metric(metric: Dict, key: str, default: float = None) -> Optional[float]:
    try:
        value = float((metric or {}).get(key, default))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _best_coverage_model(candidates: Dict, target_coverage: float = 80.0) -> str:
    """Choose interval model by calibrated target coverage, not max coverage."""
    scored = []
    for name, metric in candidates.items():
        coverage = _finite_metric(metric, "coverage_rate")
        if coverage is None:
            continue
        width = _finite_metric(metric, "mean_interval_width_pct")
        mape = _finite_metric(metric, "mape", 999999.0) or 999999.0
        if coverage >= target_coverage:
            # 覆盖率达标后，更窄的区间更有采购参考价值。
            width_score = width if width is not None else abs(coverage - target_coverage)
            score = (0, width_score, abs(coverage - target_coverage), mape)
        else:
            width_score = width if width is not None else 999999.0
            score = (1, abs(coverage - target_coverage), width_score, mape)
        scored.append((score, name))
    if not scored:
        return ""
    return min(scored, key=lambda item: item[0])[1]


def select_best_models(predictions: Dict, metrics: Dict) -> Dict[str, str]:
    """Select the best model per decision dimension."""
    candidates = _candidate_models(predictions, metrics)
    if not candidates:
        return {"price_accuracy": "", "direction": "", "coverage": ""}

    non_naive = {name: metric for name, metric in candidates.items() if name != "naive"} or candidates
    best_price = min(non_naive.items(), key=lambda item: item[1].get("mape", 999999))[0]
    direction_candidates = {
        name: metric for name, metric in non_naive.items()
        if metric.get("directional_accuracy_applicable", True) is not False
    } or non_naive
    best_direction = max(direction_candidates.items(), key=lambda item: item[1].get("directional_accuracy", -1))[0]
    best_coverage = _best_coverage_model(non_naive) or best_price
    return {
        "price_accuracy": best_price,
        "direction": best_direction,
        "coverage": best_coverage,
    }


def _direction_label(series: List[float], current_price: float) -> str:
    if not series:
        return "震荡"
    start = float(current_price) if current_price else series[0]
    end = series[min(6, len(series) - 1)]
    change_pct = (end - start) / start if start else 0.0
    if change_pct > 0.001:
        return "上涨"
    if change_pct < -0.001:
        return "下跌"
    return "震荡"


def resolve_direction_change_pct(direction: str, change_pct: float) -> Tuple[str, float]:
    """Keep the displayed direction and change percentage sign consistent."""
    if direction == "上涨":
        return direction, round(abs(float(change_pct or 0.0)), 2)
    if direction == "下跌":
        return direction, round(-abs(float(change_pct or 0.0)), 2)
    return "震荡", 0.0


def _interval_spreads(interval_pred: Dict, price_len: int) -> Tuple[List[float], List[float]]:
    p50 = _as_float_list(interval_pred.get("p50", []), price_len)
    p10 = _as_float_list(interval_pred.get("p10", []), price_len)
    p90 = _as_float_list(interval_pred.get("p90", []), price_len)
    lower = []
    upper = []
    for i in range(price_len):
        center = p50[i] if i < len(p50) else 0.0
        low = p10[i] if i < len(p10) else center * 0.99
        high = p90[i] if i < len(p90) else center * 1.01
        lower.append(max(center - low, 0.0))
        upper.append(max(high - center, 0.0))
    return lower, upper


def _normalise_segments(segments: Optional[List], horizon: int) -> List[Dict]:
    normalised = []
    for segment in segments or DEFAULT_FORECAST_SEGMENTS:
        if isinstance(segment, dict):
            key = str(segment.get("key") or segment.get("label") or "")
            start = int(segment.get("start", 0))
            end = int(segment.get("end", horizon))
        else:
            key, start, end = segment
            key = str(key)
            start = int(start)
            end = int(end)
        start = max(0, min(start, horizon))
        end = max(start, min(end, horizon))
        if key and start < end:
            normalised.append({"key": key, "start": start, "end": end})
    return normalised


def _compose_with_best(predictions: Dict, best: Dict[str, str], horizon: int) -> Tuple[List[float], List[float], List[float]]:
    price_model = best.get("price_accuracy", "")
    coverage_model = best.get("coverage") or price_model
    price_pred = (predictions or {}).get(price_model, {})
    p50 = _as_float_list(price_pred.get("p50", []), horizon)
    lower_spread, upper_spread = _interval_spreads((predictions or {}).get(coverage_model, price_pred), len(p50))
    p10 = [round(p50[i] - max(lower_spread[i], abs(p50[i]) * 0.006), 2) for i in range(len(p50))]
    p90 = [round(p50[i] + max(upper_spread[i], abs(p50[i]) * 0.006), 2) for i in range(len(p50))]
    p50 = [round(v, 2) for v in p50]
    return p50, p10, p90


def build_metric_specialized_ensemble(
    predictions: Dict,
    metrics: Dict,
    current_price: float = 0.0,
    segment_metrics: Optional[Dict[str, Dict]] = None,
    segments: Optional[List] = None,
) -> Dict:
    """Compose p50, direction and interval from the best model for each forecast segment."""
    best = select_best_models(predictions, metrics)
    price_model = best["price_accuracy"]

    if not price_model:
        return {
            "p50": [],
            "p10": [],
            "p90": [],
            "mean": [],
            "model": "综合预测",
            "best_per_metric": best,
            "best_per_segment": {},
            "direction": "震荡",
            "direction_confidence": 0.0,
        }

    price_pred = (predictions or {}).get(price_model, {})
    p50 = _as_float_list(price_pred.get("p50", []))
    horizon = len(p50)
    p50, p10, p90 = _compose_with_best(predictions, best, horizon)
    best_per_segment = {}

    if segment_metrics:
        allowed_models = set((metrics or {}).keys())
        for segment in _normalise_segments(segments, horizon):
            key = segment["key"]
            raw_seg_metrics = segment_metrics.get(key) or metrics
            seg_metrics = {name: metric for name, metric in (raw_seg_metrics or {}).items() if name in allowed_models}
            seg_best = select_best_models(predictions, seg_metrics or metrics)
            if not seg_best.get("price_accuracy"):
                continue
            seg_p50, seg_p10, seg_p90 = _compose_with_best(predictions, seg_best, horizon)
            for i in range(segment["start"], segment["end"]):
                if i < len(seg_p50):
                    p50[i] = seg_p50[i]
                    p10[i] = seg_p10[i]
                    p90[i] = seg_p90[i]
            best_per_segment[key] = {
                **seg_best,
                "start_day": segment["start"] + 1,
                "end_day": segment["end"],
            }

    direction_model = (
        best_per_segment.get("1-7天", {}).get("direction")
        or best_per_segment.get(next(iter(best_per_segment), ""), {}).get("direction")
        or best["direction"]
    )
    direction_metrics = metrics
    for seg_key, seg_metrics in (segment_metrics or {}).items():
        if best_per_segment.get(seg_key, {}).get("direction") == direction_model:
            direction_metrics = {name: metric for name, metric in (seg_metrics or {}).items() if name in metrics}
            break
    dir_series = _as_float_list((predictions or {}).get(direction_model, {}).get("p50", []), horizon)
    direction_confidence = float((direction_metrics or {}).get(direction_model, {}).get("directional_accuracy", 0.0) or 0.0)

    return {
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "mean": p50,
        "model": "综合预测",
        "best_per_metric": best_per_segment.get("1-7天", best),
        "best_per_segment": best_per_segment,
        "direction": _direction_label(dir_series, current_price),
        "direction_confidence": round(direction_confidence, 2),
    }
