"""Segment-level model evaluation for short, medium and long forecast windows."""

from typing import Dict, Iterable, List, Optional

import numpy as np

from backend.ml.baseline_models import ModelEvaluator


DEFAULT_SEGMENTS = [
    {"key": "1-7天", "start": 0, "end": 7},
    {"key": "8-14天", "start": 7, "end": 14},
    {"key": "15-30天", "start": 14, "end": 30},
]


def _as_float_array(values: Iterable) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def evaluate_forecast_segments(
    actual: Iterable,
    predictions: Dict,
    baseline_price: float,
    segments: Optional[List[Dict]] = None,
) -> Dict[str, Dict]:
    """Evaluate each model inside each forecast segment."""
    actual_arr = _as_float_array(actual)
    result: Dict[str, Dict] = {}

    for segment in segments or DEFAULT_SEGMENTS:
        key = str(segment["key"])
        start = int(segment["start"])
        end = min(int(segment["end"]), len(actual_arr))
        if start >= end:
            continue

        result[key] = {}
        actual_slice = actual_arr[start:end]
        segment_baseline = float(baseline_price) if start == 0 else float(actual_arr[start - 1])

        for model_name, pred in (predictions or {}).items():
            p50 = _as_float_array((pred or {}).get("p50", []))
            if len(p50) < end:
                continue
            p50_slice = p50[start:end]
            metrics = ModelEvaluator.evaluate(actual_slice, p50_slice, model_name)
            metrics["directional_accuracy"] = ModelEvaluator.directional_accuracy(
                actual_slice,
                p50_slice,
                baseline_actual=segment_baseline,
                baseline_predicted=segment_baseline,
            )

            p10 = _as_float_array((pred or {}).get("p10", p50))
            p90 = _as_float_array((pred or {}).get("p90", p50))
            if len(p10) >= end and len(p90) >= end:
                metrics["coverage_rate"] = ModelEvaluator.coverage_rate(actual_slice, p10[start:end], p90[start:end])
                metrics["coverage_rate_applicable"] = True
            else:
                metrics["coverage_rate"] = None
                metrics["coverage_rate_applicable"] = False

            result[key][model_name] = metrics

    return result
