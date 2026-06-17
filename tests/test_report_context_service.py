import unittest

from pydantic import ValidationError

from backend.models.schemas import StructuredAnalysisReport
from backend.services.report_context_service import (
    build_forecast_evidence_bundle,
    collect_evidence_ids,
    missing_report_citations,
    validate_report_citations,
)


def _report(cited_evidence_ids=None, adjustment_proposal=None):
    return StructuredAnalysisReport(
        summary="Short-term forecast remains stable.",
        trend_view="P50 points imply a mild upward bias over the forecast horizon.",
        procurement_advice={"action": "stage_orders", "reasoning": "Use lower interval bids."},
        risk_flags=["news sentiment may shift demand expectations"],
        confidence=0.72,
        assumptions=["No direct rewrite of model forecast points."],
        cited_evidence_ids=cited_evidence_ids or [],
        model_limitations=["Backtest window is short."],
        adjustment_proposal=adjustment_proposal,
    )


class ReportContextServiceTests(unittest.TestCase):
    def test_minimal_cache_builds_evidence_bundle(self):
        cache = {
            "current_price": 100.0,
            "predictions": {
                "ensemble": {
                    "p50": [101.0, 102.0],
                    "p10": [99.0, 100.0],
                    "p90": [103.0, 104.0],
                    "qa_passed": True,
                    "qa_summary": "Layer 1 QA passed",
                }
            },
            "metrics": {
                "ensemble": {"mape": 1.2, "directional_accuracy": 75.0, "coverage_rate": 90.0}
            },
        }

        bundle = build_forecast_evidence_bundle(cache, commodity="diesel_0", as_of_date="2026-06-03")

        self.assertEqual(bundle.commodity, "diesel_0")
        self.assertEqual(bundle.as_of_date, "2026-06-03")
        self.assertEqual(bundle.current_price, 100.0)
        self.assertEqual(bundle.prediction_horizon, 2)
        self.assertEqual(len(bundle.model_evidence), 1)
        self.assertEqual(bundle.model_evidence[0].model_name, "ensemble")
        self.assertEqual(bundle.model_evidence[0].prediction_summary["p50_7d"], [101.0, 102.0])
        self.assertEqual(bundle.qa_summary[0].value["passed"], True)

    def test_evidence_ids_are_stable_and_report_citations_are_validatable(self):
        cache = {
            "current_price": 100.0,
            "model_disagreement": True,
            "predictions": {
                "ensemble": {
                    "p50": [101.0, 102.0, 103.0],
                    "p10": [99.0, 100.0, 101.0],
                    "p90": [103.0, 104.0, 105.0],
                    "qa_passed": True,
                    "qa_summary": "All checks passed",
                    "best_per_metric": {"price_accuracy": "xgboost", "direction": "prophet"},
                }
            },
            "metrics": {
                "ensemble": {"mape": 1.0, "directional_accuracy": 80.0, "coverage_rate": 95.0}
            },
            "segment_metrics": {
                "day_1_2": {
                    "ensemble": {"mape": 0.8, "directional_accuracy": 100.0, "coverage_rate": 100.0}
                }
            },
            "news_sentiment": {
                "summary": "Supply news is supportive.",
                "sentiment_score": 0.4,
                "price_adjustment_pct": 0.002,
                "news_items": [
                    {
                        "title": "Refinery maintenance tightens supply",
                        "sentiment": "bullish",
                        "impact_score": 0.7,
                        "source_url": "https://example.com/refinery-maintenance",
                    }
                ],
            },
            "data_quality": {"completeness": 98.0, "outlier_pct": 1.0},
            "fixed_split_evaluation": {"status": "ready", "models": [{"model_name": "ensemble"}]},
        }

        first = build_forecast_evidence_bundle(cache, commodity="diesel_0", as_of_date="2026-06-03")
        second = build_forecast_evidence_bundle(cache, commodity="diesel_0", as_of_date="2026-06-03")

        first_ids = collect_evidence_ids(first)
        second_ids = collect_evidence_ids(second)
        self.assertEqual(first_ids, second_ids)
        self.assertTrue(first.model_evidence[0].segment_evidence_ids)
        self.assertEqual(first.news_evidence[0].impact, "Supply news is supportive.")

        cited_ids = [
            first.model_evidence[0].evidence_id,
            next(iter(first.model_evidence[0].segment_evidence_ids.values())),
            first.qa_summary[0].evidence_id,
            first.news_evidence[0].evidence_id,
            first.fixed_split_metrics.evidence_id,
            first.data_quality.evidence_id,
            first.ensemble_rationale.evidence_id,
        ]
        report = _report(
            cited_evidence_ids=cited_ids,
            adjustment_proposal={
                "recommendation": "Keep model p50 unchanged; review a small sentiment bias.",
                "suggested_bias_pct": 0.2,
                "cited_evidence_ids": [first.news_evidence[0].evidence_id],
            },
        )

        self.assertTrue(validate_report_citations(report, first))
        self.assertEqual(missing_report_citations(report, first), [])

        bad_report = _report(cited_evidence_ids=["missing:evidence"])
        self.assertFalse(validate_report_citations(bad_report, first))
        self.assertEqual(missing_report_citations(bad_report, first), ["missing:evidence"])

        bad_nested_report = _report(
            adjustment_proposal={
                "recommendation": "Review analyst bias only.",
                "cited_evidence_ids": ["missing:nested-evidence"],
            }
        )
        self.assertFalse(validate_report_citations(bad_nested_report, first))
        self.assertEqual(
            missing_report_citations(bad_nested_report, first),
            ["missing:nested-evidence"],
        )

    def test_adjustment_proposal_is_advisory_not_forecast_rewrite(self):
        report = _report(
            adjustment_proposal={
                "recommendation": "Review a small sentiment bias during analyst sign-off.",
                "suggested_bias_pct": -0.1,
                "rationale": "News impact is weak and should not overwrite model outputs.",
            }
        )

        self.assertFalse(hasattr(report.adjustment_proposal, "p50"))
        proposal_payload = (
            report.adjustment_proposal.model_dump()
            if hasattr(report.adjustment_proposal, "model_dump")
            else report.adjustment_proposal.dict()
        )
        self.assertNotIn("p50", proposal_payload)

        with self.assertRaises(ValidationError):
            _report(
                adjustment_proposal={
                    "recommendation": "Rewrite model forecast directly.",
                    "p50": [99.0, 98.0],
                }
            )


if __name__ == "__main__":
    unittest.main()
