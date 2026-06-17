import asyncio
import json
import unittest

from backend.models.schemas import EvidenceItem, ForecastEvidenceBundle, ModelEvidence
from backend.services.llm_service import LLMService
from backend.services.report_context_service import (
    missing_report_citations,
    validate_report_citations,
)


def _minimal_bundle():
    return ForecastEvidenceBundle(
        commodity="diesel_0",
        as_of_date="2026-06-03",
        current_price=100.0,
        prediction_horizon=3,
        model_evidence=[
            ModelEvidence(
                evidence_id="model:ensemble",
                model_name="ensemble",
                prediction_summary={
                    "p50_7d": [100.0, 101.0, 102.0],
                    "p10_7d": [98.0, 99.0, 100.0],
                    "p90_7d": [102.0, 103.0, 104.0],
                },
                metrics={"mape": 1.2, "directional_accuracy": 80.0},
            )
        ],
        risk_flags=[
            EvidenceItem(
                evidence_id="risk:qa",
                source="qa_engine",
                title="QA passed with limited horizon",
                value={"passed": True, "horizon": 3},
                confidence=0.8,
            )
        ],
    )


def _valid_llm_payload(**overrides):
    payload = {
        "summary": "LLM report should only be accepted when it cites evidence.",
        "trend_view": "The cited model evidence shows p50 rising over the horizon.",
        "procurement_advice": {
            "action": "hold_or_stage",
            "reasoning": "Use staged execution while monitoring risk evidence.",
            "timing": "Review before the next purchase window.",
        },
        "risk_flags": ["QA horizon is limited."],
        "confidence": 0.8,
        "assumptions": ["Forecast values are unchanged."],
        "cited_evidence_ids": ["model:ensemble"],
        "model_limitations": ["Short horizon."],
        "adjustment_proposal": None,
    }
    payload.update(overrides)
    return payload


def _generate_report_from_payload(payload):
    class PayloadLLM(LLMService):
        def _init_clients(self):
            self.client = None

        def _call_llm(self, *args, **kwargs):
            return json.dumps(payload)

    return asyncio.run(PayloadLLM().generate_structured_analysis_report(_minimal_bundle()))


class StructuredReportLLMTests(unittest.TestCase):
    def test_no_llm_client_generates_fallback_structured_report(self):
        bundle = _minimal_bundle()
        service = LLMService()

        report = asyncio.run(service.generate_structured_analysis_report(bundle))

        self.assertTrue(validate_report_citations(report, bundle))
        self.assertEqual(missing_report_citations(report, bundle), [])
        self.assertIn("model:ensemble", report.cited_evidence_ids)
        self.assertTrue(report.risk_flags)
        self.assertTrue(report.model_limitations)
        self.assertIsNone(report.adjustment_proposal)

    def test_missing_llm_citations_are_removed_and_confidence_is_lowered(self):
        class InvalidCitationLLM(LLMService):
            def _init_clients(self):
                self.client = None

            def _call_llm(self, *args, **kwargs):
                return json.dumps({
                    "summary": "Forecast has a mild upward bias.",
                    "trend_view": "The cited model evidence shows p50 rising over the horizon.",
                    "procurement_advice": {
                        "action": "hold_or_stage",
                        "reasoning": "Use staged execution while monitoring risk evidence.",
                        "timing": "Review before the next purchase window.",
                    },
                    "risk_flags": ["QA horizon is limited."],
                    "confidence": 0.8,
                    "assumptions": ["Forecast values are unchanged."],
                    "cited_evidence_ids": ["model:ensemble", "missing:top"],
                    "model_limitations": ["Short horizon."],
                    "adjustment_proposal": {
                        "recommendation": "Review a small analyst bias.",
                        "suggested_bias_pct": 0.1,
                        "rationale": "QA risk should be reviewed before action.",
                        "cited_evidence_ids": ["missing:nested", "risk:qa"],
                        "review_required": True,
                    },
                })

        bundle = _minimal_bundle()
        service = InvalidCitationLLM()

        report = asyncio.run(service.generate_structured_analysis_report(bundle))

        self.assertTrue(validate_report_citations(report, bundle))
        self.assertEqual(missing_report_citations(report, bundle), [])
        self.assertEqual(report.cited_evidence_ids, ["model:ensemble"])
        self.assertEqual(report.adjustment_proposal.cited_evidence_ids, ["risk:qa"])
        self.assertLess(report.confidence, 0.8)

    def test_adjustment_proposal_cannot_rewrite_p50(self):
        class P50RewriteLLM(LLMService):
            def _init_clients(self):
                self.client = None

            def _call_llm(self, *args, **kwargs):
                return json.dumps({
                    "summary": "Forecast should be rewritten.",
                    "trend_view": "The report attempts to overwrite the forecast.",
                    "procurement_advice": {
                        "action": "buy",
                        "reasoning": "This should be rejected.",
                        "timing": "now",
                    },
                    "risk_flags": [],
                    "confidence": 0.9,
                    "assumptions": [],
                    "cited_evidence_ids": ["model:ensemble"],
                    "model_limitations": [],
                    "adjustment_proposal": {
                        "recommendation": "Directly replace model p50.",
                        "p50": [99.0, 98.0, 97.0],
                        "cited_evidence_ids": ["model:ensemble"],
                    },
                })

        bundle = _minimal_bundle()
        service = P50RewriteLLM()

        report = asyncio.run(service.generate_structured_analysis_report(bundle))

        self.assertTrue(validate_report_citations(report, bundle))
        self.assertIsNone(report.adjustment_proposal)
        self.assertLessEqual(report.confidence, 0.45)
        self.assertIn("p10/p50/p90", " ".join(report.model_limitations + report.assumptions))

    def test_empty_or_cleaned_top_citations_fall_back_to_valid_evidence(self):
        for cited_evidence_ids in ([], ["missing:top"]):
            with self.subTest(cited_evidence_ids=cited_evidence_ids):
                report = _generate_report_from_payload(
                    _valid_llm_payload(cited_evidence_ids=cited_evidence_ids)
                )

                self.assertTrue(validate_report_citations(report, _minimal_bundle()))
                self.assertIn("model:ensemble", report.cited_evidence_ids)
                self.assertIn("deterministic fallback", " ".join(report.model_limitations))

    def test_top_level_forecast_keys_are_rejected_before_schema_ignores_them(self):
        for forbidden_key, value in (
            ("p50", [99.0, 100.0, 101.0]),
            ("predictions", {"p50": [99.0, 100.0, 101.0]}),
        ):
            with self.subTest(forbidden_key=forbidden_key):
                report = _generate_report_from_payload(
                    _valid_llm_payload(**{forbidden_key: value})
                )

                self.assertTrue(validate_report_citations(report, _minimal_bundle()))
                self.assertIn("model:ensemble", report.cited_evidence_ids)
                self.assertIn("deterministic fallback", " ".join(report.model_limitations))

    def test_adjustment_proposal_without_valid_citations_is_not_trusted(self):
        report = _generate_report_from_payload(
            _valid_llm_payload(
                adjustment_proposal={
                    "recommendation": "Review a small analyst bias.",
                    "suggested_bias_pct": 0.1,
                    "rationale": "This proposal has no supported citation.",
                    "cited_evidence_ids": [],
                    "review_required": True,
                }
            )
        )

        self.assertTrue(validate_report_citations(report, _minimal_bundle()))
        self.assertIn("model:ensemble", report.cited_evidence_ids)
        self.assertIsNone(report.adjustment_proposal)


if __name__ == "__main__":
    unittest.main()
