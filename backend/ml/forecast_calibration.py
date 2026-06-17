"""Forecast volatility calibration utilities."""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


_ACTUAL_VALUE_KEYS = ("actual", "actual_price", "target", "y", "value", "close", "price")
_PREDICTION_VALUE_KEYS = ("p50", "prediction", "predicted", "yhat", "forecast", "center", "value")
_HORIZON_KEYS = ("horizon", "h", "step", "day", "period")


def _float_array(values: Iterable) -> np.ndarray:
    result = []
    if values is None:
        return np.asarray(result, dtype=float)
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(number):
        return number
    return None


def _is_dataframe_like(data: Any) -> bool:
    return hasattr(data, "to_dict") and hasattr(data, "columns")


def _records(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if _is_dataframe_like(data):
        try:
            return [dict(row) for row in data.to_dict(orient="records")]
        except TypeError:
            return []
    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, (str, bytes)):
        return [{"value": data}]

    try:
        if isinstance(data, np.ndarray):
            iterable = data.tolist() if data.ndim else [data.item()]
        else:
            iterable = list(data)
    except TypeError:
        iterable = [data]

    rows = []
    for item in iterable:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            rows.append({"value": item})
    return rows


def _first_finite(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            number = _safe_float(row.get(key))
            if number is not None:
                return number
    return None


def _normalize_horizon(value: Any, fallback: Any = None) -> Any:
    number = _safe_float(value)
    if number is not None:
        return int(number) if number.is_integer() else number

    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _extract_horizon(row: Mapping[str, Any], horizon_key: Optional[str] = None) -> Any:
    keys = []
    if horizon_key:
        keys.append(horizon_key)
    keys.extend(key for key in _HORIZON_KEYS if key not in keys)

    for key in keys:
        if key in row:
            horizon = _normalize_horizon(row.get(key))
            if horizon is not None:
                return horizon
    return None


def _horizon_values(horizons: Any) -> List[Any]:
    values = []
    for row in _records(horizons):
        horizon = _extract_horizon(row)
        if horizon is None and "value" in row:
            horizon = _normalize_horizon(row.get("value"))
        values.append(horizon)
    return values


def _conservative_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return 0.0
    try:
        return float(np.quantile(array, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(array, quantile, interpolation="higher"))


def _clamped_probability(value: Any, default: float) -> float:
    number = _safe_float(value)
    if number is None:
        return default
    return min(max(number, 0.0), 1.0)


def _lookup_residual_quantile(residual_quantiles: Mapping[Any, Any], horizon: Any) -> Optional[float]:
    if not isinstance(residual_quantiles, Mapping):
        return None

    candidates = [horizon, str(horizon)]
    number = _safe_float(horizon)
    if number is not None:
        candidates.extend([int(number) if number.is_integer() else number, float(number)])

    seen = set()
    for candidate in candidates:
        try:
            if candidate in seen:
                continue
            seen.add(candidate)
        except TypeError:
            pass
        if candidate in residual_quantiles:
            return _safe_float(residual_quantiles[candidate])

    for fallback_key in ("overall", "global", "__global__", "all"):
        if fallback_key in residual_quantiles:
            return _safe_float(residual_quantiles[fallback_key])
    return None


def _interval_target_coverage(lower_key: str, upper_key: str, fallback: float = 0.8) -> float:
    lower_match = re.fullmatch(r"p(\d+(?:\.\d+)?)", str(lower_key).lower())
    upper_match = re.fullmatch(r"p(\d+(?:\.\d+)?)", str(upper_key).lower())
    if not lower_match or not upper_match:
        return fallback

    lower = _safe_float(lower_match.group(1))
    upper = _safe_float(upper_match.group(1))
    if lower is None or upper is None or upper <= lower:
        return fallback
    return min(max((upper - lower) / 100.0, 0.0), 1.0)


def _round_list(values: np.ndarray) -> List[float]:
    return [round(float(v), 2) for v in values]


def _repeat_to_length(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(length, dtype=float)
    repeats = int(np.ceil(length / len(values)))
    return np.tile(values, repeats)[:length]


def calculate_horizon_residual_quantiles(
    actuals: Any,
    predictions: Any,
    horizons: Any = None,
    quantile: float = 0.9,
) -> Dict[Any, float]:
    """
    Calculate conservative absolute residual quantiles grouped by forecast horizon.

    Inputs may be numeric sequences, list[dict], or pandas-like DataFrames. Rows
    with missing, non-finite, or unparseable actual/prediction values are ignored.
    """
    actual_rows = _records(actuals)
    prediction_rows = _records(predictions)
    explicit_horizons = _horizon_values(horizons)
    if not actual_rows or not prediction_rows:
        return {}

    probability = _clamped_probability(quantile, 0.9)
    residuals_by_horizon: Dict[Any, List[float]] = {}

    for index in range(min(len(actual_rows), len(prediction_rows))):
        actual = _first_finite(actual_rows[index], _ACTUAL_VALUE_KEYS)
        prediction = _first_finite(prediction_rows[index], _PREDICTION_VALUE_KEYS)
        if actual is None or prediction is None:
            continue

        horizon = explicit_horizons[index] if index < len(explicit_horizons) else None
        if horizon is None:
            horizon = _extract_horizon(actual_rows[index])
        if horizon is None:
            horizon = _extract_horizon(prediction_rows[index])
        if horizon is None:
            horizon = index + 1

        residuals_by_horizon.setdefault(horizon, []).append(abs(actual - prediction))

    return {
        horizon: _conservative_quantile(values, probability)
        for horizon, values in residuals_by_horizon.items()
        if values
    }


def apply_conformal_intervals(
    forecast_points: Any,
    residual_quantiles: Mapping[Any, Any],
    lower_key: str = "p10",
    center_key: str = "p50",
    upper_key: str = "p90",
    horizon_key: str = "horizon",
) -> Any:
    """
    Widen forecast intervals using per-horizon residual quantiles.

    The center forecast is preserved. Existing intervals are widened only when the
    conformal residual requires it, and output bounds are kept monotonic around
    the center.
    """
    records = _records(forecast_points)
    if not records:
        if _is_dataframe_like(forecast_points):
            return forecast_points.copy()
        return []

    def calibrated_row(row: Mapping[str, Any], index: int) -> Dict[str, Any]:
        updated = dict(row)
        center = _safe_float(row.get(center_key))
        if center is None:
            return updated

        horizon = _extract_horizon(row, horizon_key)
        if horizon is None:
            horizon = index + 1
        residual = _lookup_residual_quantile(residual_quantiles, horizon)

        lower = _safe_float(row.get(lower_key))
        upper = _safe_float(row.get(upper_key))
        if residual is None and lower is None and upper is None:
            return updated

        if residual is not None:
            residual = max(residual, 0.0)
            conformal_lower = center - residual
            conformal_upper = center + residual
            lower = conformal_lower if lower is None else min(lower, conformal_lower)
            upper = conformal_upper if upper is None else max(upper, conformal_upper)

        if lower is None or upper is None:
            return updated

        if lower > upper:
            lower, upper = upper, lower

        updated[lower_key] = float(min(lower, center))
        updated[upper_key] = float(max(upper, center))
        return updated

    calibrated_records = [calibrated_row(row, index) for index, row in enumerate(records)]

    if _is_dataframe_like(forecast_points):
        result = forecast_points.copy()
        indexes = list(result.index)
        for index, row in enumerate(calibrated_records):
            if lower_key in row:
                result.at[indexes[index], lower_key] = row[lower_key]
            if upper_key in row:
                result.at[indexes[index], upper_key] = row[upper_key]
        return result

    return calibrated_records


def evaluate_interval_coverage(
    actuals: Any,
    forecast_points: Any,
    lower_key: str = "p10",
    upper_key: str = "p90",
    horizon_key: str = "horizon",
    target_coverage: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate interval coverage overall and by horizon.

    coverage_gap is target coverage minus observed coverage, so positive values
    indicate under-coverage.
    """
    actual_rows = _records(actuals)
    forecast_rows = _records(forecast_points)
    target = (
        _clamped_probability(target_coverage, 0.8)
        if target_coverage is not None
        else _interval_target_coverage(lower_key, upper_key)
    )

    groups: Dict[Any, Dict[str, Any]] = {}
    for index in range(min(len(actual_rows), len(forecast_rows))):
        actual = _first_finite(actual_rows[index], _ACTUAL_VALUE_KEYS)
        lower = _safe_float(forecast_rows[index].get(lower_key))
        upper = _safe_float(forecast_rows[index].get(upper_key))
        if actual is None or lower is None or upper is None:
            continue

        if lower > upper:
            lower, upper = upper, lower

        horizon = _extract_horizon(forecast_rows[index], horizon_key)
        if horizon is None:
            horizon = _extract_horizon(actual_rows[index], horizon_key)
        if horizon is None:
            horizon = index + 1

        stats = groups.setdefault(horizon, {"covered": 0, "count": 0, "widths": []})
        stats["covered"] += int(lower <= actual <= upper)
        stats["count"] += 1
        stats["widths"].append(upper - lower)

    total_count = sum(stats["count"] for stats in groups.values())
    total_covered = sum(stats["covered"] for stats in groups.values())
    all_widths = [width for stats in groups.values() for width in stats["widths"]]

    coverage_rate = (total_covered / total_count) if total_count else 0.0
    mean_width = float(np.mean(all_widths)) if all_widths else 0.0

    per_horizon = {}
    for horizon, stats in groups.items():
        horizon_coverage = stats["covered"] / stats["count"] if stats["count"] else 0.0
        horizon_width = float(np.mean(stats["widths"])) if stats["widths"] else 0.0
        per_horizon[horizon] = {
            "coverage_rate": horizon_coverage,
            "coverage_gap": target - horizon_coverage,
            "mean_interval_width": horizon_width,
            "count": stats["count"],
        }

    return {
        "coverage_rate": coverage_rate,
        "coverage_gap": target - coverage_rate,
        "mean_interval_width": mean_width,
        "target_coverage": target,
        "count": total_count,
        "per_horizon": per_horizon,
    }


def calibrate_forecast_volatility(
    p50: Iterable,
    p10: Iterable,
    p90: Iterable,
    historical_prices: Iterable,
    min_step_ratio: float = 0.35,
) -> Dict[str, List[float]]:
    """
    Add a deterministic historical-volatility shape when a forecast is too flat.

    The goal is not to invent a new trend. It preserves the first and last forecast
    level while restoring day-to-day movement from recent historical changes.
    """
    mid = _float_array(p50)
    low = _float_array(p10)
    high = _float_array(p90)
    history = _float_array(historical_prices)

    horizon = len(mid)
    if horizon < 3 or len(history) < 8:
        return {"p50": _round_list(mid), "p10": _round_list(low), "p90": _round_list(high)}

    if len(low) != horizon:
        low = mid - np.maximum(mid * 0.01, 1.0)
    if len(high) != horizon:
        high = mid + np.maximum(mid * 0.01, 1.0)

    recent = history[-min(len(history), 90):]
    changes = np.diff(recent)
    changes = changes[np.isfinite(changes)]
    if len(changes) < 4:
        return {"p50": _round_list(mid), "p10": _round_list(low), "p90": _round_list(high)}

    hist_step_std = float(np.std(changes))
    hist_abs_p75 = float(np.percentile(np.abs(changes), 75))
    target_step_std = max(hist_step_std * min_step_ratio, hist_abs_p75 * 0.22, float(mid[0]) * 0.0015)
    current_step_std = float(np.std(np.diff(mid))) if horizon > 1 else 0.0

    if current_step_std >= target_step_std * 0.8:
        return {"p50": _round_list(mid), "p10": _round_list(low), "p90": _round_list(high)}

    template_changes = _repeat_to_length(changes[-max(4, min(len(changes), horizon * 2)):], horizon - 1)
    template_changes = template_changes - float(np.mean(template_changes))
    template_std = float(np.std(template_changes))
    if template_std <= 1e-9:
        return {"p50": _round_list(mid), "p10": _round_list(low), "p90": _round_list(high)}

    template_changes = template_changes / template_std * target_step_std
    offsets = np.concatenate([[0.0], np.cumsum(template_changes)])

    # Remove net drift so the selected model still controls the forecast trend.
    offsets = offsets - np.linspace(offsets[0], offsets[-1], horizon)
    calibrated_mid = mid + offsets
    shift = calibrated_mid - mid

    calibrated_low = low + shift
    calibrated_high = high + shift
    spread_floor = np.maximum(np.abs(calibrated_mid) * 0.006, target_step_std * 0.8)
    calibrated_low = np.minimum(calibrated_low, calibrated_mid - spread_floor)
    calibrated_high = np.maximum(calibrated_high, calibrated_mid + spread_floor)

    return {
        "p50": _round_list(calibrated_mid),
        "p10": _round_list(calibrated_low),
        "p90": _round_list(calibrated_high),
    }
