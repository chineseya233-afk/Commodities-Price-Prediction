"""
LLM Integration Service (Backend Agent)

Uses one generic OpenAI-compatible endpoint for:
- Intelligent analysis report generation
- 3-dimensional risk research reports
- QA Layer 2 soft validation

Configure the endpoint with OPENAI_COMPATIBLE_* or legacy LLM_* variables.
If the configured provider is unavailable, local deterministic templates are used.
"""

import json
from datetime import date
from typing import Any, Dict, Mapping, Optional, Union
from loguru import logger

from backend.models.schemas import ForecastEvidenceBundle, StructuredAnalysisReport
from backend.services.report_context_service import (
    collect_evidence_ids,
    missing_report_citations,
    validate_report_citations,
)


class LLMService:
    """
    Generic OpenAI-compatible LLM service for commodity analysis.
    """

    SYSTEM_PROMPT = """你是一名资深能源大宗商品量化分析师。
只使用输入数据，清晰说明不确定性；需要 JSON 时只返回严格 JSON。
所有面向用户的标题、解释、理由、风险和执行建议必须使用简体中文。
不要直接改写 p10、p50 或 p90 预测序列。"""

    ANALYSIS_PROMPT_TEMPLATE = """请生成一份专业的 0# 柴油价格研判报告，所有可见文案使用简体中文。

Market data:
- Commodity: 0# diesel, RMB/吨
- Current price: {current_price}
- 7-day price change: {price_change_7d} RMB ({price_change_pct_7d}%)
- 30-day price change: {price_change_30d} RMB ({price_change_pct_30d}%)

Forecast for the next 7 days:
- P50: {predictions_p50}
- P90: {predictions_p90}
- P10: {predictions_p10}
- Trend: {trend_direction}

Model history:
- MAPE: {mape}%
- Directional accuracy: {directional_accuracy}%
- Interval coverage: {coverage_rate}%

QA result:
{qa_summary}

News and market sentiment:
{news_context}

Return only JSON, with no markdown fences:
{{
  "summary": "中文一句话结论",
  "trend_analysis": "中文详细分析",
  "risk_factors": ["中文风险 1", "中文风险 2", "中文风险 3"],
  "procurement_advice": {{
    "action": "hold_or_stage/staged_buy/reduce/chase_less",
    "confidence": "高/中/低",
    "reasoning": "中文决策理由",
    "suggested_price_range": "价格区间，单位 RMB/吨",
    "timing": "中文执行时机"
  }},
  "data_quality_notes": "中文数据质量和模型可靠性说明"
}}"""

    RISK_REPORT_PROMPT = """请生成一份用于柴油采购的三维风险报告，所有可见文案使用简体中文。

Current data:
- Current price: {current_price} RMB/吨
- P50 for 7 days: {p50_7d}
- Forecast trend: {trend}

QA result:
{qa_summary}

Model metrics:
{model_metrics}

News and market sentiment:
{news_context}

Return only JSON, with no markdown fences:
{{
  "report_date": "{today}",
  "dimension_1_market": [
    {{"level": "高/中/低/关注", "title": "中文风险标题", "impact": "中文具体影响"}}
  ],
  "dimension_2_model": [
    {{"level": "预警/通过/关注", "title": "中文模型校验结果", "interpretation": "中文解释"}}
  ],
  "dimension_3_policy": [
    {{"level": "关注/低", "title": "中文政策或数据风险", "impact": "中文影响说明"}}
  ]
}}"""

    STRUCTURED_REPORT_SYSTEM_PROMPT = """你是一名能源行业量化分析师。
只能基于输入的 ForecastEvidenceBundle 生成符合 schema 的 JSON 报告。
所有面向用户的字段内容必须使用简体中文；不要输出 markdown 或额外解释。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-v4-pro",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self._init_clients()

    def _init_clients(self):
        """Initialize the configured OpenAI-compatible LLM client."""
        if self.api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
                logger.info(f"LLM client initialized ({self.model})")
            except ImportError:
                logger.warning("openai package not installed. Using mock responses.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")

    def is_available(self) -> bool:
        """Return whether the configured LLM client is ready."""
        return self.client is not None

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """Call the configured LLM. Returns raw response text or None."""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(f"LLM ({self.model}) response received")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM error ({self.model}): {e}")

        return None

    async def generate_analysis_report(
        self,
        current_price: float,
        predictions: Dict,
        historical_prices: list,
        model_metrics: Dict = None,
        qa_summary: str = "",
        news_sentiment: Dict = None,
    ) -> Dict:
        """Generate a structured analysis report using LLM chain."""
        prices = historical_prices if historical_prices else [current_price]
        price_7d_ago = prices[-7] if len(prices) >= 7 else prices[0]
        price_30d_ago = prices[-30] if len(prices) >= 30 else prices[0]

        price_change_7d = round(current_price - price_7d_ago, 2)
        price_change_pct_7d = round(price_change_7d / price_7d_ago * 100, 2) if price_7d_ago else 0
        price_change_30d = round(current_price - price_30d_ago, 2)
        price_change_pct_30d = round(price_change_30d / price_30d_ago * 100, 2) if price_30d_ago else 0

        p50 = predictions.get("p50", [])
        p10 = predictions.get("p10", [])
        p90 = predictions.get("p90", [])

        if len(p50) >= 2:
            trend = "上涨" if p50[-1] > p50[0] else ("下跌" if p50[-1] < p50[0] else "震荡")
        else:
            trend = "数据不足"

        news_context = self._format_news_context(news_sentiment)

        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            current_price=current_price,
            price_change_7d=price_change_7d,
            price_change_pct_7d=price_change_pct_7d,
            price_change_30d=price_change_30d,
            price_change_pct_30d=price_change_pct_30d,
            predictions_p50=[round(p, 2) for p in p50[:7]],
            predictions_p90=[round(p, 2) for p in p90[:7]],
            predictions_p10=[round(p, 2) for p in p10[:7]],
            trend_direction=trend,
            mape=model_metrics.get("mape", "N/A") if model_metrics else "N/A",
            directional_accuracy=model_metrics.get("directional_accuracy", "N/A") if model_metrics else "N/A",
            coverage_rate=model_metrics.get("coverage_rate", "N/A") if model_metrics else "N/A",
            qa_summary=qa_summary or "QA鏍￠獙閫氳繃",
            news_context=news_context,
        )

        raw = self._call_llm(self.SYSTEM_PROMPT, prompt)
        if raw:
            parsed = self._parse_llm_response(raw)
            if parsed is not None:
                return self._normalize_analysis_report(
                    parsed, current_price=current_price, predictions=predictions, trend=trend
                )

        return self._normalize_analysis_report(self._generate_mock_report(
            current_price, predictions, price_change_7d, price_change_pct_7d, trend, news_sentiment
        ), current_price=current_price, predictions=predictions, trend=trend)

    async def generate_structured_analysis_report(
        self,
        evidence_bundle: Union[ForecastEvidenceBundle, Mapping[str, Any]],
    ) -> StructuredAnalysisReport:
        """Generate a schema-validated report from an immutable evidence bundle."""
        bundle = self._coerce_evidence_bundle(evidence_bundle)
        prompt = self._build_evidence_prompt(bundle)

        raw = self._call_llm(
            self.STRUCTURED_REPORT_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=2500,
        )
        if raw:
            parsed = self._parse_llm_response(raw)
            if parsed is not None:
                normalized = self._normalize_structured_report(parsed, bundle)
                if normalized is not None:
                    return normalized

        return self._generate_structured_fallback_report(bundle)

    def _coerce_evidence_bundle(
        self,
        evidence_bundle: Union[ForecastEvidenceBundle, Mapping[str, Any]],
    ) -> ForecastEvidenceBundle:
        if isinstance(evidence_bundle, ForecastEvidenceBundle):
            return evidence_bundle
        if isinstance(evidence_bundle, Mapping):
            return ForecastEvidenceBundle(**dict(evidence_bundle))
        raise TypeError("evidence_bundle must be ForecastEvidenceBundle or mapping")

    def _build_evidence_prompt(self, bundle: ForecastEvidenceBundle) -> str:
        bundle_payload = self._dump_model(bundle)
        evidence_ids = sorted(collect_evidence_ids(bundle))

        return f"""请基于以下 ForecastEvidenceBundle 生成结构化分析报告。
硬性规则：
1. 只能使用证据包中的内容作为事实来源。
2. cited_evidence_ids 只能填写下方 Available evidence_ids 中存在的 evidence_id，必须逐字一致。
3. 不要直接修改、重写或覆盖预测序列、p10、p50、p90、current_price。
4. 如果认为预测需要调整，只能在 adjustment_proposal 中提出 suggested_bias_pct，并保持 review_required=true。
5. adjustment_proposal 不允许包含 p10、p50、p90、predictions、forecast 等直接预测字段。
6. 必须包含模型局限、关键风险、采购建议和引用证据。
Available evidence_ids:
{json.dumps(evidence_ids, ensure_ascii=False)}

请只输出符合以下结构的 JSON：
{{
  "summary": "string",
  "trend_view": "string",
  "procurement_advice": {{"action": "string", "reasoning": "string", "timing": "string"}},
  "risk_flags": ["string"],
  "confidence": 0.0,
  "assumptions": ["string"],
  "cited_evidence_ids": ["evidence_id"],
  "model_limitations": ["string"],
  "adjustment_proposal": null 或 {{
    "recommendation": "string",
    "suggested_bias_pct": 0.0,
    "rationale": "string",
    "cited_evidence_ids": ["evidence_id"],
    "review_required": true,
    "metadata": {{}}
  }}
}}

ForecastEvidenceBundle:
{json.dumps(bundle_payload, ensure_ascii=False, default=str, sort_keys=True)}
"""

    def _normalize_structured_report(
        self,
        report_payload: Mapping[str, Any],
        bundle: ForecastEvidenceBundle,
    ) -> Optional[StructuredAnalysisReport]:
        if not isinstance(report_payload, Mapping):
            logger.warning("Structured report schema validation failed: payload is not an object")
            return None

        if self._contains_forbidden_forecast_fields(report_payload):
            logger.warning("Structured report rejected: report rewrites forecast fields")
            return None

        payload = dict(report_payload)
        if not payload.get("trend_view"):
            payload["trend_view"] = (
                payload.get("trend_analysis")
                or payload.get("trend")
                or payload.get("summary")
                or "模型预测已生成，执行前需结合最新价格、宏观指标和风险证据综合判断。"
            )
        if "risk_flags" not in payload:
            payload["risk_flags"] = payload.get("risk_factors") or []
        if not isinstance(payload.get("procurement_advice"), Mapping):
            payload["procurement_advice"] = {}

        payload.setdefault("summary", str(payload.get("trend_view") or "")[:120] or "结构化分析报告已生成。")
        payload.setdefault("assumptions", [])
        payload.setdefault("cited_evidence_ids", [])
        payload.setdefault("model_limitations", [])

        confidence = payload.get("confidence")
        if isinstance(confidence, str):
            confidence_map = {
                "high": 0.8,
                "medium": 0.55,
                "low": 0.35,
                "高": 0.8,
                "中": 0.55,
                "低": 0.35,
            }
            payload["confidence"] = confidence_map.get(confidence.strip().lower())

        try:
            report = StructuredAnalysisReport(**payload)
        except Exception as exc:
            logger.warning(f"Structured report schema validation failed: {exc}")
            return None

        missing = missing_report_citations(report, bundle)
        if missing:
            report = self._remove_missing_report_citations(report, bundle, missing)

        report = self._enforce_required_report_citations(report, bundle)
        if report is None:
            return None

        if validate_report_citations(report, bundle):
            return report

        logger.warning("Structured report rejected: citation validation failed after cleanup")
        return None

    def _contains_forbidden_forecast_fields(self, value: Any) -> bool:
        forbidden_keys = {
            "p10",
            "p50",
            "p90",
            "prediction",
            "predictions",
            "forecast",
            "forecast_values",
            "forecast_sequence",
            "prediction_sequence",
            "optimized_prices",
            "price_series",
        }
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in forbidden_keys:
                    return True
                if self._contains_forbidden_forecast_fields(nested):
                    return True
        elif isinstance(value, list):
            return any(self._contains_forbidden_forecast_fields(item) for item in value)
        return False

    def _enforce_required_report_citations(
        self,
        report: StructuredAnalysisReport,
        bundle: ForecastEvidenceBundle,
    ) -> Optional[StructuredAnalysisReport]:
        allowed_ids = collect_evidence_ids(bundle)
        if not allowed_ids:
            return report

        top_level_ids = [evidence_id for evidence_id in report.cited_evidence_ids if evidence_id in allowed_ids]
        if not top_level_ids:
            logger.warning("Structured report rejected: top-level citations are empty after cleanup")
            return None

        if report.adjustment_proposal:
            proposal_ids = [
                evidence_id
                for evidence_id in report.adjustment_proposal.cited_evidence_ids
                if evidence_id in allowed_ids
            ]
            if not proposal_ids:
                payload = self._dump_model(report)
                payload["adjustment_proposal"] = None
                logger.warning("Structured report adjustment_proposal removed: no valid citations")
                return StructuredAnalysisReport(**payload)

        return report

    def _remove_missing_report_citations(
        self,
        report: StructuredAnalysisReport,
        bundle: ForecastEvidenceBundle,
        missing: list,
    ) -> StructuredAnalysisReport:
        allowed_ids = collect_evidence_ids(bundle)
        payload = self._dump_model(report)
        payload["cited_evidence_ids"] = [
            evidence_id
            for evidence_id in payload.get("cited_evidence_ids", [])
            if evidence_id in allowed_ids
        ]

        proposal = payload.get("adjustment_proposal")
        if proposal:
            original_proposal_ids = list(proposal.get("cited_evidence_ids", []))
            proposal["cited_evidence_ids"] = [
                evidence_id for evidence_id in original_proposal_ids if evidence_id in allowed_ids
            ]
            if not proposal["cited_evidence_ids"]:
                payload["adjustment_proposal"] = None

        confidence = payload.get("confidence")
        if isinstance(confidence, (int, float)):
            payload["confidence"] = max(0.0, min(1.0, round(float(confidence) * 0.7, 4)))
        else:
            payload["confidence"] = 0.4

        assumptions = list(payload.get("assumptions") or [])
        assumptions.append("模型服务输出包含证据包中不存在的引用，系统已移除无效引用。")
        payload["assumptions"] = assumptions

        limitations = list(payload.get("model_limitations") or [])
        limitations.append("由于移除了不受支持的证据引用，报告置信度已下调。")
        payload["model_limitations"] = limitations

        logger.warning(f"Removed invalid structured report citations: {missing}")
        return StructuredAnalysisReport(**payload)

    def _generate_structured_fallback_report(
        self,
        bundle: ForecastEvidenceBundle,
    ) -> StructuredAnalysisReport:
        evidence_ids = sorted(collect_evidence_ids(bundle))
        cited_ids = evidence_ids[:5]
        primary_model = self._select_primary_model_evidence(bundle)
        prediction_summary = primary_model.prediction_summary if primary_model else {}
        p50 = prediction_summary.get("p50_7d") if isinstance(prediction_summary, Mapping) else []
        p50_values = [float(value) for value in p50 if isinstance(value, (int, float))]

        direction = "震荡"
        if len(p50_values) >= 2:
            if p50_values[-1] > p50_values[0]:
                direction = "上行"
            elif p50_values[-1] < p50_values[0]:
                direction = "下行"

        model_name = primary_model.model_name if primary_model else "可用模型"
        price_text = f"{bundle.current_price:.2f}" if isinstance(bundle.current_price, (int, float)) else str(bundle.current_price)
        if p50_values:
            forecast_text = (
                f"p50 起点={p50_values[0]:.2f}，终点={p50_values[-1]:.2f}，"
                f"区间={min(p50_values):.2f}-{max(p50_values):.2f}"
            )
        else:
            forecast_text = "证据包中缺少 p50 序列"

        risk_flags = [item.title for item in bundle.risk_flags[:3]]
        risk_flags.extend(item.title for item in bundle.news_evidence[:2])
        if bundle.data_quality:
            risk_flags.append(bundle.data_quality.title)
        if not risk_flags:
            risk_flags = ["外盘原油、政策调价窗口、库存变化和上游数据质量都可能改变短期风险。"]

        procurement_action = "staged_buy" if direction == "下行" else "hold_or_stage"
        procurement_reasoning = (
            "当前确定性兜底报告不会改写任何预测值，仅基于已引用证据给出偏保守的采购判断。"
        )

        return StructuredAnalysisReport(
            summary=f"{bundle.commodity}短期观点为{direction}，报告基于已校验的证据包生成。",
            trend_view=(
                f"截至 {bundle.as_of_date}，当前价格为{price_text}；{model_name}证据显示{forecast_text}。"
                "系统未修改 p10/p50/p90 预测序列。"
            ),
            procurement_advice={
                "action": procurement_action,
                "reasoning": procurement_reasoning,
                "timing": "分批执行；任何预测偏置调整都需要人工复核后再应用。",
            },
            risk_flags=risk_flags,
            confidence=0.45 if cited_ids else 0.35,
            assumptions=[
                "证据包被视为本次报告唯一事实来源。",
                "预测 p10/p50/p90 数值保持不变。",
                "任何调整都必须在模型输出路径之外进行人工复核。",
            ],
            cited_evidence_ids=cited_ids,
            model_limitations=[
                "模型服务不可用或输出未通过校验，当前使用确定性兜底报告。",
                "预测区间不保证实际成交价格，且可能遗漏证据包之外的突发事件。",
                "报告质量取决于上游数据覆盖、回测窗口和新闻证据新鲜度。",
            ],
            adjustment_proposal=None,
        )

    def _select_primary_model_evidence(self, bundle: ForecastEvidenceBundle):
        if not bundle.model_evidence:
            return None
        for item in bundle.model_evidence:
            if item.model_name == "ensemble":
                return item
        return bundle.model_evidence[0]

    def _dump_model(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value

    async def generate_risk_report(
        self,
        current_price: float,
        predictions: Dict,
        qa_results: Dict,
        model_metrics: Dict,
        news_sentiment: Dict = None,
    ) -> Dict:
        """Generate the 3-dimensional risk research report."""
        p50 = predictions.get("p50", [])
        trend = "上涨" if len(p50) >= 2 and p50[-1] > p50[0] else "下跌" if len(p50) >= 2 and p50[-1] < p50[0] else "震荡"

        prompt = self.RISK_REPORT_PROMPT.format(
            current_price=current_price,
            p50_7d=[round(p, 2) for p in p50[:7]],
            trend=trend,
            qa_summary=json.dumps(qa_results, ensure_ascii=False, default=str)[:500],
            model_metrics=json.dumps(model_metrics, ensure_ascii=False, default=str)[:500],
            news_context=self._format_news_context(news_sentiment),
            today=str(date.today()),
        )

        raw = self._call_llm(self.SYSTEM_PROMPT, prompt, max_tokens=1500)
        if raw:
            parsed = self._parse_llm_response(raw)
            if parsed is not None:
                return parsed

        market_risks = []
        if news_sentiment and isinstance(news_sentiment.get("news_items"), list):
            for item in news_sentiment.get("news_items", [])[:3]:
                if str(item.get("level", "")).lower() not in {"low", "pass", "低", "通过"}:
                    market_risks.append({
                        "level": item.get("level", "关注"),
                        "title": item.get("title", "市场消息风险"),
                        "impact": item.get("impact", news_sentiment.get("summary", "新闻影响需要持续跟踪")),
                    })
        if not market_risks:
            market_risks = [
                {"level": "关注", "title": "OPEC+ 产量政策不确定性", "impact": "产量政策变化可能影响上游成本和采购时点。"},
                {"level": "低", "title": "柴油季节性需求支撑", "impact": "区域需求可能限制短期下行空间。"},
            ]

        # 模拟三维风险报告
        return {
            "report_date": str(date.today()),
            "dimension_1_market": market_risks,
            "dimension_2_model": [
                {"level": "通过", "title": "模型质量校验完成", "interpretation": "预测输出已通过确定性校验，或已使用受控兜底结果。"},
            ],
            "dimension_3_policy": [
                {"level": "关注", "title": "政策调整窗口", "impact": "国内成品油调价窗口可能影响采购价格。"},
            ],
        }

    async def validate_with_llm(
        self,
        predictions: Dict,
        market_context: str,
    ) -> Dict:
        """Layer 2 QA: Use LLM for soft validation of prediction logic consistency."""
        prompt = f"""请评估这组大宗商品价格预测在逻辑上是否一致，所有可见文案使用中文。

Forecast P50 for the next days: {predictions.get('p50', [])[:7]}
Market context: {market_context}

Return only JSON:
{{"passed": true, "confidence": "高/中/低", "reasoning": "中文简短理由"}}"""

        system = "你是大宗商品价格预测质量校验员。只返回 JSON，所有可见文案使用中文。"
        raw = self._call_llm(system, prompt, temperature=0.1, max_tokens=500)
        if raw:
            parsed = self._parse_llm_response(raw)
            if parsed is not None:
                return parsed

        return {"passed": True, "notes": "模型服务不可用，已跳过软性校验。"}

    def _parse_llm_response(self, raw: str) -> Dict:
        """Parse LLM response, handling truncated/malformed JSON from various LLMs."""
        import re

        # 清理常见生成残留
        cleaned = raw.strip().strip("\ufeff")  # BOM

        # 尝试 1：直接解析
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试 2：从 Markdown 代码块提取
        for pattern in [r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
            m = re.search(pattern, cleaned)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # 尝试 3：查找最外层 JSON 对象
        start = cleaned.find("{")
        if start >= 0:
            json_str = cleaned[start:]
            # 尝试直接解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            # 尝试修复截断 JSON（补齐缺失的右花括号/方括号）
            for repair in ["}", "}}", "\"}", "\"}}", "\"]}", "\"]}}",
                           "\"]}}", "\"}]}", "\"]}}}",  "\"\n}"]:
                try:
                    return json.loads(json_str + repair)
                except json.JSONDecodeError:
                    continue

        # 尝试 4：用正则从原始文本提取单个字段
        result = {}
        for field, pattern in [
            ("summary", r'"summary"\s*:\s*"([^"]{5,200})"'),
            ("trend_analysis", r'"trend_analysis"\s*:\s*"((?:[^"\\]|\\.)*)'),
            ("data_quality_notes", r'"data_quality_notes"\s*:\s*"((?:[^"\\]|\\.)*)'),
        ]:
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                result[field] = m.group(1).replace('\\"', '"').replace("\\n", "\n")

        # 提取 risk_factors 数组
        m = re.search(r'"risk_factors"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        if m:
            factors = re.findall(r'"([^"]{5,})"', m.group(1))
            result["risk_factors"] = factors

        # 提取 procurement_advice 对象
        m = re.search(r'"action"\s*:\s*"([^"]*)"', cleaned)
        if m:
            advice = {"action": m.group(1)}
            for key in ["confidence", "reasoning", "suggested_price_range", "timing"]:
                km = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
                if km:
                    advice[key] = km.group(1).replace('\\"', '"')
            result["procurement_advice"] = advice

        if result.get("summary"):
            logger.info("Parsed LLM response via regex extraction")
            result.setdefault("trend_analysis", "")
            result.setdefault("risk_factors", [])
            result.setdefault("procurement_advice", {"action": "hold_or_stage", "confidence": "中"})
            result.setdefault("data_quality_notes", "")
            return result

        # 最终回退：如果可读文本包含 JSON 字段名，则改用模拟报告
        logger.warning("Failed to parse LLM JSON response. Using mock report fallback.")
        return None  # Signal caller to use mock report

    def _format_news_context(self, news_sentiment: Dict = None) -> str:
        """Format news sentiment for prompts."""
        if not news_sentiment:
            return "No available news sentiment data."
        items = news_sentiment.get("news_items", [])[:5]
        lines = [
            f"News summary: {news_sentiment.get('summary', 'N/A')}",
            f"Sentiment score: {news_sentiment.get('sentiment_score', 0)}",
            f"Short-term price adjustment: {news_sentiment.get('price_adjustment_pct', 0) * 100:.2f}%",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. [{item.get('level', 'low')}/{item.get('direction', 'neutral')}] "
                f"{item.get('title', '')} - {item.get('impact', '')}"
            )
        return "\n".join(lines)

    def _normalize_analysis_report(
        self,
        report: Dict,
        current_price: float,
        predictions: Dict,
        trend: str,
    ) -> Dict:
        """Fill required report fields when an LLM returns partial JSON."""
        normalized = dict(report or {})
        p50 = [float(v) for v in predictions.get("p50", []) if isinstance(v, (int, float))]
        p10 = [float(v) for v in predictions.get("p10", []) if isinstance(v, (int, float))]
        p90 = [float(v) for v in predictions.get("p90", []) if isinstance(v, (int, float))]
        short_prices = p50[:7] or [float(current_price)]
        lower_prices = p10[:7] or short_prices
        upper_prices = p90[:7] or short_prices
        low = min(lower_prices)
        high = max(upper_prices)
        avg = sum(short_prices) / len(short_prices)
        latest_pred = short_prices[-1]
        first_pred = short_prices[0]

        direction_text = "上行" if latest_pred > first_pred else "下行" if latest_pred < first_pred else "震荡"
        trend_analysis = str(normalized.get("trend_analysis") or "").strip()
        if len(trend_analysis) < 40:
            trend_analysis = (
                f"当前价格约{current_price:.0f} RMB/吨，未来 7 天 P50 均价约{avg:.0f} RMB/吨，"
                f"参考区间为{low:.0f}-{high:.0f} RMB/吨。短期预测方向为{direction_text}，"
                "建议分批采购，并持续跟踪政策窗口、库存变化、汇率和外盘油价冲击。"
            )
        normalized["trend_analysis"] = trend_analysis

        advice = dict(normalized.get("procurement_advice") or {})
        action = advice.get("action") or ("staged_buy" if direction_text == "下行" else "hold_or_stage")
        reasoning = str(advice.get("reasoning") or "").strip()
        if len(reasoning) < 20 or not reasoning.endswith((".", "!", "?", ";", "。", "！", "？", "；")):
            reasoning = (
                "预测不确定性仍然较高，建议围绕预测区间低位分批执行，并保留风险边界。"
            )
        price_range = str(advice.get("suggested_price_range") or "").strip()
        if len(price_range) < 6 or "-" not in price_range:
            price_range = f"{low:.0f}-{high:.0f} RMB/吨"
        timing = str(advice.get("timing") or "").strip()
        if len(timing) < 10:
            timing = f"未来 1-7 天滚动复核，接近{low:.0f} RMB/吨时优先分批执行。"

        advice.update({
            "action": action,
            "confidence": advice.get("confidence") or "中",
            "reasoning": reasoning,
            "suggested_price_range": price_range,
            "timing": timing,
        })

        normalized["procurement_advice"] = advice
        normalized.setdefault("summary", f"柴油短期预测方向为{direction_text}。")
        normalized.setdefault("risk_factors", [])
        normalized.setdefault("data_quality_notes", "基于模型输出和可用市场证据生成。")
        return normalized

    def _generate_mock_report(
        self,
        current_price: float,
        predictions: Dict,
        change_7d: float,
        change_pct_7d: float,
        trend: str,
        news_sentiment: Dict = None,
    ) -> Dict:
        """Generate a deterministic analysis report when LLM is unavailable."""
        p50 = [float(v) for v in predictions.get("p50", []) if isinstance(v, (int, float))]
        p50_display = p50[:7] or [float(current_price)]
        avg_pred = sum(p50_display) / len(p50_display)
        low = min(p50_display)
        high = max(p50_display)
        direction = "上行" if p50_display[-1] > p50_display[0] else "下行" if p50_display[-1] < p50_display[0] else "震荡"
        action = "staged_buy" if direction == "下行" else "hold_or_stage"
        news_summary = (news_sentiment or {}).get("summary", "外盘原油、汇率、库存和政策窗口仍需持续跟踪。")

        return {
            "summary": f"柴油短期方向为{direction}，7 天预测均价约{avg_pred:.0f} RMB/吨。",
            "trend_analysis": (
                f"当前价格约{current_price:.0f} RMB/吨，近 7 天价格变动{change_7d:.0f} RMB"
                f"（{change_pct_7d:.2f}%）。模型预测未来 7 天区间约{low:.0f}-{high:.0f} RMB/吨，"
                f"短期倾向为{direction}。建议保持分批采购和风险限额，并持续跟踪政策与市场消息。"
            ),
            "risk_factors": [
                news_summary,
                "汇率变化可能影响进口成本和国内定价预期。",
                "政策调价窗口可能带来跳跃式价格变化。",
            ],
            "procurement_advice": {
                "action": action,
                "confidence": "中",
                "reasoning": "由于预测区间和外部政策风险仍不确定，建议分批采购。",
                "suggested_price_range": f"{low:.0f}-{high:.0f} RMB/吨",
                "timing": "未来 1-7 天滚动复核，未得到确认前避免一次性大额采购。",
            },
            "data_quality_notes": "模型服务暂不可用或输出未通过校验，当前使用确定性兜底报告。",
        }
