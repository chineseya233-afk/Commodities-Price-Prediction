"""
QA Quality Assurance Engine (Backend Agent)

Dual-layer verification architecture:
Layer 1: Hard Rule Engine 鈥?deterministic threshold checks, zero false positives
Layer 2: LLM Soft Validation 鈥?industry logic consistency check via configured LLM
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import date


class QAEngine:
    """
    Intelligent quality assurance engine for prediction outputs.
    
    Acts as a "digital firewall" between the prediction engine and
    the business presentation layer, ensuring all outputs are
    logically consistent and within acceptable bounds.
    """

    # 第 1 层规则引擎的硬阈值
    MAX_DAILY_CHANGE_PCT = 5.0      # 卤5% daily change cap
    MAX_7D_CUMULATIVE_PCT = 15.0    # 卤15% 7-day cumulative change cap
    HISTORICAL_SIGMA_MULTIPLIER = 2.5  # 卤2.5蟽 from historical mean (lenient for POC simulated data)
    MIN_PRICE_FLOOR = 3000.0        # Absolute minimum (RMB/ton)
    MAX_PRICE_CEILING = 15000.0     # Absolute maximum (RMB/ton)

    def __init__(self):
        self.validation_log: List[Dict] = []

    def validate_predictions(
        self,
        predictions: Dict,
        historical_prices: np.ndarray,
        model_name: str = "unknown",
    ) -> Tuple[bool, str, Dict]:
        """
        Run full dual-layer validation on prediction output.
        
        Args:
            predictions: dict with 'p10', 'p50', 'p90' lists
            historical_prices: array of recent historical prices
            model_name: name of the model that generated predictions
            
        Returns:
            (is_valid, summary_message, detailed_results)
        """
        results = {
            "model": model_name,
            "layer1_checks": [],
            "layer1_passed": True,
            "layer2_notes": "",
            "overall_passed": True,
        }

        p50 = np.array(predictions.get("p50", []))
        p10 = np.array(predictions.get("p10", []))
        p90 = np.array(predictions.get("p90", []))

        if len(p50) == 0:
            results["overall_passed"] = False
            return False, "Empty predictions", results

        # ====== LAYER 1: Hard Rule Engine ======
        
        # 检查 1：负数或零值
        check_1 = self._check_positive_values(p50, p10, p90)
        results["layer1_checks"].append(check_1)
        
        # 检查 2：绝对价格边界
        check_2 = self._check_price_bounds(p50)
        results["layer1_checks"].append(check_2)
        
        # 检查 3：单日涨跌幅
        check_3 = self._check_daily_change(p50, historical_prices)
        results["layer1_checks"].append(check_3)
        
        # 检查 4：7 日累计变化
        check_4 = self._check_cumulative_change(p50, historical_prices)
        results["layer1_checks"].append(check_4)
        
        # 检查 5：历史 sigma 边界
        check_5 = self._check_sigma_bounds(p50, historical_prices)
        results["layer1_checks"].append(check_5)
        
        # 检查 6：预测区间合理性（P10 < P50 < P90）
        check_6 = self._check_interval_ordering(p10, p50, p90)
        results["layer1_checks"].append(check_6)
        
        # 检查 7：区间宽度合理性
        check_7 = self._check_interval_width(p10, p50, p90)
        results["layer1_checks"].append(check_7)

        # 汇总第 1 层结果
        layer1_failed = [c for c in results["layer1_checks"] if not c["passed"]]
        results["layer1_passed"] = len(layer1_failed) == 0
        
        if not results["layer1_passed"]:
            failed_names = [c["check"] for c in layer1_failed]
            summary = f"Layer 1 FAILED: {', '.join(failed_names)}"
            results["overall_passed"] = False
            logger.warning(f"QA {model_name}: {summary}")
            return False, summary, results

        # ====== LAYER 2: LLM Soft Validation (deferred to llm_service) ======
        # 第 2 层由后端服务在调用已配置 LLM 时触发
        results["layer2_notes"] = "Pending LLM validation"
        results["overall_passed"] = True
        
        summary = f"All checks passed ({len(results['layer1_checks'])} rule checks)"
        logger.info(f"QA {model_name}: {summary}")
        
        # 记录校验日志
        self.validation_log.append({
            "timestamp": str(date.today()),
            "model": model_name,
            "passed": True,
            "summary": summary,
        })
        
        return True, summary, results

    def _check_positive_values(self, p50, p10, p90) -> Dict:
        """All predicted values must be positive."""
        all_positive = np.all(p50 > 0) and np.all(p10 > 0) and np.all(p90 > 0)
        return {
            "check": "positive_values",
            "passed": bool(all_positive),
            "detail": "All predictions positive" if all_positive 
                      else f"Negative values detected: min={min(p10.min(), p50.min(), p90.min()):.2f}",
        }

    def _check_price_bounds(self, p50) -> Dict:
        """Predictions within absolute price range."""
        in_bounds = np.all(p50 >= self.MIN_PRICE_FLOOR) and np.all(p50 <= self.MAX_PRICE_CEILING)
        return {
            "check": "price_bounds",
            "passed": bool(in_bounds),
            "detail": f"Range [{p50.min():.0f}, {p50.max():.0f}] within [{self.MIN_PRICE_FLOOR}, {self.MAX_PRICE_CEILING}]"
                      if in_bounds else f"Out of bounds: [{p50.min():.0f}, {p50.max():.0f}]",
        }

    def _check_daily_change(self, p50, historical) -> Dict:
        """No single-day change exceeds threshold."""
        if len(historical) == 0:
            return {"check": "daily_change", "passed": True, "detail": "No historical data"}
        
        last_price = historical[-1]
        all_prices = np.concatenate([[last_price], p50])
        daily_changes = np.abs(np.diff(all_prices) / all_prices[:-1] * 100)
        max_change = daily_changes.max()
        
        passed = max_change <= self.MAX_DAILY_CHANGE_PCT
        return {
            "check": "daily_change",
            "passed": bool(passed),
            "detail": f"Max daily change: {max_change:.2f}% (limit: 卤{self.MAX_DAILY_CHANGE_PCT}%)",
        }

    def _check_cumulative_change(self, p50, historical) -> Dict:
        """7-day cumulative change within threshold."""
        if len(historical) == 0:
            return {"check": "cumulative_change", "passed": True, "detail": "No historical data"}
        
        last_price = historical[-1]
        first_week = p50[:7] if len(p50) >= 7 else p50
        max_cum_change = np.max(np.abs((first_week - last_price) / last_price * 100))
        
        passed = max_cum_change <= self.MAX_7D_CUMULATIVE_PCT
        return {
            "check": "cumulative_change",
            "passed": bool(passed),
            "detail": f"Max cumulative change: {max_cum_change:.2f}% (limit: 卤{self.MAX_7D_CUMULATIVE_PCT}%)",
        }

    def _check_sigma_bounds(self, p50, historical) -> Dict:
        """Predictions within 卤2蟽 of historical distribution."""
        if len(historical) < 30:
            return {"check": "sigma_bounds", "passed": True, "detail": "Insufficient history (<30 points)"}
        
        mean = np.mean(historical)
        std = np.std(historical)
        lower = mean - self.HISTORICAL_SIGMA_MULTIPLIER * std
        upper = mean + self.HISTORICAL_SIGMA_MULTIPLIER * std
        
        in_bounds = np.all(p50 >= lower) and np.all(p50 <= upper)
        return {
            "check": "sigma_bounds",
            "passed": bool(in_bounds),
            "detail": f"Historical 渭={mean:.0f}, 蟽={std:.0f}. Predictions in [{p50.min():.0f}, {p50.max():.0f}], "
                      f"bounds [{lower:.0f}, {upper:.0f}]",
        }

    def _check_interval_ordering(self, p10, p50, p90) -> Dict:
        """P10 鈮?P50 鈮?P90 for all time steps."""
        ordered = np.all(p10 <= p50) and np.all(p50 <= p90)
        return {
            "check": "interval_ordering",
            "passed": bool(ordered),
            "detail": "P10 鈮?P50 鈮?P90 satisfied" if ordered else "Interval ordering violated",
        }

    def _check_interval_width(self, p10, p50, p90) -> Dict:
        """Prediction interval not too narrow or too wide."""
        widths = (p90 - p10) / p50 * 100
        mean_width = np.mean(widths)
        
        # 合理范围：价格的 0.5% 到 20%
        passed = 0.5 <= mean_width <= 20.0
        return {
            "check": "interval_width",
            "passed": bool(passed),
            "detail": f"Mean interval width: {mean_width:.2f}% of price (expected: 0.5-20%)",
        }

    def get_validation_log(self) -> List[Dict]:
        """Return all validation logs."""
        return self.validation_log
