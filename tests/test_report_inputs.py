import unittest

from backend import main
from backend.services.llm_service import LLMService


class ReportInputTests(unittest.TestCase):
    def test_analysis_report_fills_incomplete_procurement_advice(self):
        class PartialAdviceLLM(LLMService):
            def _init_clients(self):
                self.client = None

            def _call_llm(self, *args, **kwargs):
                return """
                {
                  "summary": "短期价格承压，建议谨慎采购。",
                  "trend_analysis": "近期柴油价格呈现",
                  "risk_factors": ["库存波动风险", "调价窗口风险"],
                  "procurement_advice": {
                    "action": "逢低加仓",
                    "confidence": "中",
                    "reasoning": "模型预测短期价格下跌，但新闻情绪偏利多提供",
                    "suggested_price_range": "62",
                    "timing": ""
                  }
                }
                """

        service = PartialAdviceLLM()

        import asyncio
        report = asyncio.run(service.generate_analysis_report(
            current_price=6318.33,
            predictions={"p50": [6327.25, 6262.37, 6303.79, 6293.02, 6293.64, 6295.0, 6307.22]},
            historical_prices=[6200, 6210, 6220, 6230, 6240, 6250, 6318.33],
            model_metrics={"mape": 0.35, "directional_accuracy": 71.4, "coverage_rate": 100},
            news_sentiment={"summary": "新闻情绪偏利多", "price_adjustment_pct": 0.0023},
        ))

        advice = report["procurement_advice"]
        self.assertEqual(advice["action"], "逢低加仓")
        self.assertGreaterEqual(len(report["trend_analysis"]), 40)
        self.assertGreaterEqual(len(advice["reasoning"]), 20)
        self.assertTrue(advice["reasoning"].endswith(("。", "！", "？", ".", "!", "?")))
        self.assertNotIn("利多提供", advice["reasoning"])
        self.assertIn("suggested_price_range", advice)
        self.assertIn("timing", advice)
        self.assertNotEqual(advice["suggested_price_range"], "N/A")
        self.assertIn("-", advice["suggested_price_range"])
        self.assertNotEqual(advice["suggested_price_range"], "62")

    def test_report_regeneration_prefers_ensemble_and_news_context(self):
        original_predictions = main._cache.get("predictions")
        original_metrics = main._cache.get("metrics")
        original_news = main._cache.get("news_sentiment")
        try:
            main._cache["predictions"] = {
                "prophet": {"p50": [100.0]},
                "ensemble": {"p50": [101.0], "news_sentiment_applied": True},
            }
            main._cache["metrics"] = {
                "prophet": {"mape": 0.5},
                "ensemble": {"mape": 1.0, "directional_accuracy": 80.0},
            }
            main._cache["news_sentiment"] = {"summary": "新闻情绪偏利多"}

            prediction, metrics, news = main._select_report_inputs()

            self.assertIs(prediction, main._cache["predictions"]["ensemble"])
            self.assertIs(metrics, main._cache["metrics"]["ensemble"])
            self.assertEqual(news["summary"], "新闻情绪偏利多")
        finally:
            main._cache["predictions"] = original_predictions
            main._cache["metrics"] = original_metrics
            main._cache["news_sentiment"] = original_news


if __name__ == "__main__":
    unittest.main()
