"""
News sentiment service for diesel price risk adjustment.

POC design:
- Try public RSS feeds for oil/energy headlines.
- Fall back to deterministic market events when feeds are unavailable.
- Use LLM analysis when possible, with a bounded rule-based scorer as backup.
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx
import numpy as np
from loguru import logger


class NewsSentimentService:
    """Fetch and score market news, returning a bounded forecast adjustment."""

    RSS_FEEDS = [
        ("中国能源网", "https://www.zgnyw.net/mobile/feed/"),
        ("EIA", "https://www.eia.gov/rss/todayinenergy.xml"),
        ("OilPrice", "https://oilprice.com/rss/main"),
        ("Reuters Energy", "https://feeds.reuters.com/reuters/USenergyNews"),
    ]

    HTML_SOURCES = [
        (
            "国家发改委成品油价格",
            "https://www.ndrc.gov.cn/xwdt/ztzl/gncpyjg/wap_index.html",
            ["成品油", "柴油", "汽油", "价格"],
        ),
        (
            "国家统计局生产资料价格",
            "https://www.stats.gov.cn/sj/zxfb/",
            ["流通领域", "生产资料", "成品油", "柴油"],
        ),
        (
            "中国能源新闻网油气",
            "https://www.cpnn.com.cn/news/yq/",
            ["柴油", "成品油", "原油", "油气", "炼化"],
        ),
        (
            "中国能源网",
            "https://www.cnenergynews.cn/",
            ["柴油", "成品油", "原油", "油气", "炼化"],
        ),
    ]

    BULLISH_PATTERNS = {
        "supply cut": 2.0,
        "production cut": 2.0,
        "opec cut": 2.0,
        "sanction": 1.5,
        "conflict": 1.3,
        "inventory draw": 1.5,
        "lower inventories": 1.5,
        "inventories fall": 1.5,
        "demand rises": 1.4,
        "demand growth": 1.2,
        "prices supported": 1.0,
        "costs supported": 1.0,
        "crude prices rise": 1.1,
        "减产": 2.0,
        "供应收紧": 1.8,
        "库存下降": 1.5,
        "需求回升": 1.3,
        "地缘冲突": 1.3,
        "油价上涨": 1.1,
    }

    BEARISH_PATTERNS = {
        "supply increase": -1.8,
        "production increase": -1.7,
        "inventory build": -1.5,
        "inventories rise": -1.5,
        "demand weak": -1.4,
        "demand falls": -1.4,
        "recession": -1.2,
        "crude prices fall": -1.1,
        "增产": -1.8,
        "供应增加": -1.7,
        "库存上升": -1.5,
        "需求疲弱": -1.4,
        "经济衰退": -1.2,
        "油价下跌": -1.1,
    }

    def __init__(self, fetch_timeout: float = 2.5):
        self.fetch_timeout = fetch_timeout

    async def get_market_sentiment(
        self,
        llm_service=None,
        current_price: float = 0.0,
        limit: int = 8,
    ) -> Dict:
        """Fetch market news and return LLM or rule-based sentiment."""
        news = await self.fetch_news(limit=limit)

        llm_attempted = False
        if llm_service and (
            llm_service.is_available()
            if hasattr(llm_service, "is_available")
            else getattr(llm_service, "client", None)
        ):
            try:
                llm_attempted = True
                llm_result = self.analyze_with_llm(llm_service, news, current_price)
                if llm_result:
                    llm_result.setdefault("source", "llm")
                    return llm_result
            except Exception as exc:
                logger.warning(f"News LLM sentiment failed, using rules: {exc}")

        result = self.analyze_rule_based(news, current_price=current_price)
        if llm_attempted:
            result["llm_attempted"] = True
        if (
            abs(result.get("sentiment_score", 0.0)) < 0.05
            and not any(item.get("level") != "低" for item in result.get("news_items", []))
        ):
            fallback_result = self.analyze_rule_based(self._fallback_news(), current_price=current_price)
            fallback_result["source"] = "poc_market_events"
            fallback_result["llm_attempted"] = llm_attempted
            fallback_result["live_news_items"] = result.get("news_items", [])
            return fallback_result
        result["source"] = "rules"
        return result

    async def fetch_news(self, limit: int = 8) -> List[Dict]:
        """Fetch RSS headlines. Return fallback market events if no feed works."""
        items: List[Dict] = []
        async with httpx.AsyncClient(timeout=self.fetch_timeout, follow_redirects=True) as client:
            for source, url in self.RSS_FEEDS:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    items.extend(self._parse_rss(response.text, source, limit=limit))
                    if len(items) >= limit:
                        break
                except Exception as exc:
                    logger.debug(f"News feed unavailable ({source}): {exc}")

            if len(items) < limit:
                for source, url, keywords in self.HTML_SOURCES:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        items.extend(self._parse_html_links(response.text, source, url, keywords, limit=limit))
                        if len(items) >= limit:
                            break
                    except Exception as exc:
                        logger.debug(f"China market source unavailable ({source}): {exc}")

        if not items:
            items = self._fallback_news()

        return items[:limit]

    def _parse_rss(self, raw_xml: str, source: str, limit: int) -> List[Dict]:
        root = ET.fromstring(raw_xml)
        parsed: List[Dict] = []
        for item in root.findall(".//item")[:limit]:
            title = self._text(item, "title")
            summary = self._strip_html(self._text(item, "description"))
            published_at = self._text(item, "pubDate")
            link = self._text(item, "link")
            if title:
                parsed.append({
                    "id": f"{source}:{abs(hash(title))}",
                    "title": title,
                    "summary": summary,
                    "published_at": published_at,
                    "source": source,
                    "url": link,
                })
        return parsed

    def _text(self, item, tag: str) -> str:
        value = item.findtext(tag)
        return value.strip() if value else ""

    def _strip_html(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()

    def _parse_html_links(
        self,
        raw_html: str,
        source: str,
        base_url: str,
        keywords: List[str],
        limit: int,
    ) -> List[Dict]:
        parsed: List[Dict] = []
        seen = set()
        link_pattern = re.compile(
            r"<a\b[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
            re.IGNORECASE | re.DOTALL,
        )
        for match in link_pattern.finditer(raw_html or ""):
            title = self._strip_html(match.group("title"))
            title = re.sub(r"\s+", " ", title).strip()
            if len(title) < 4:
                continue
            if keywords and not any(keyword in title for keyword in keywords):
                continue
            url = urljoin(base_url, match.group("href"))
            key = (source, title)
            if key in seen:
                continue
            seen.add(key)
            parsed.append({
                "id": f"{source}:{abs(hash(title))}",
                "title": title,
                "summary": title,
                "published_at": str(date.today()),
                "source": source,
                "url": url,
            })
            if len(parsed) >= limit:
                break
        return parsed

    def _fallback_news(self) -> List[Dict]:
        today = str(date.today())
        return [
            {
                "id": "poc:opec-policy",
                "title": "OPEC+产量政策仍是成品油成本的关键变量",
                "summary": "若主要产油国维持减产纪律，原油输入成本短期可能获得支撑，国内柴油采购锁价压力上升。",
                "published_at": today,
                "source": "POC市场监测",
                "url": "",
            },
            {
                "id": "poc:inventory",
                "title": "柴油库存变化可能放大区域价格波动",
                "summary": "库存下降叠加物流和工矿需求恢复时，国内柴油现货报价更容易上行；库存回补则会压制涨幅。",
                "published_at": today,
                "source": "POC市场监测",
                "url": "",
            },
            {
                "id": "poc:fx-policy",
                "title": "汇率与发改委调价窗口影响采购执行时点",
                "summary": "人民币汇率波动和成品油调价窗口可能改变到岸成本与终端报价节奏，需要在采购执行前复核。",
                "published_at": today,
                "source": "POC市场监测",
                "url": "",
            },
        ]

    def analyze_rule_based(self, news: List[Dict], current_price: float = 0.0) -> Dict:
        """Score headlines with a transparent keyword model."""
        scored_items = []
        weighted_scores = []

        for item in news:
            raw_score = self._score_text(f"{item.get('title', '')} {item.get('summary', '')}")
            normalized = float(np.clip(raw_score / 5.0, -1.0, 1.0))
            if abs(raw_score) >= 6:
                level = "高"
            elif abs(raw_score) >= 2:
                level = "中"
            else:
                level = "低"
            direction = "利多" if raw_score > 0 else "利空" if raw_score < 0 else "中性"
            weighted_scores.append(normalized)
            scored_items.append({
                **item,
                "sentiment": round(normalized, 3),
                "direction": direction,
                "level": level,
                "impact": self._impact_text(direction, level),
            })

        sentiment_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))
        price_adjustment_pct = round(sentiment_score * 0.01, 5)
        if 0.05 < abs(sentiment_score) < 0.2:
            price_adjustment_pct = round(0.0025 if sentiment_score > 0 else -0.0025, 5)

        return {
            "as_of": str(date.today()),
            "sentiment_score": round(sentiment_score, 4),
            "price_adjustment_pct": price_adjustment_pct,
            "impact_direction": "up" if sentiment_score > 0.05 else "down" if sentiment_score < -0.05 else "neutral",
            "confidence": "中" if abs(sentiment_score) >= 0.2 else "低",
            "news_items": scored_items,
            "summary": self._summary(sentiment_score, price_adjustment_pct),
            "applied": False,
        }

    def analyze_with_llm(self, llm_service, news: List[Dict], current_price: float) -> Optional[Dict]:
        """Ask LLM for sentiment, then normalize and bound the result."""
        payload = [
            {
                "id": item.get("id", str(idx)),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
            }
            for idx, item in enumerate(news[:8], start=1)
        ]
        prompt = f"""请分析以下柴油/原油相关新闻对未来7-30天国内0号柴油价格的影响。

当前柴油价格: {current_price:.0f} RMB/吨
新闻列表:
{json.dumps(payload, ensure_ascii=False)}

请严格输出JSON，不要Markdown：
{{
  "sentiment_score": -1到1之间的小数,
  "price_adjustment_pct": -0.012到0.012之间的小数,
  "impact_direction": "up/down/neutral",
  "confidence": "高/中/低",
  "summary": "一句话说明新闻如何影响价格",
  "news_items": [
    {{"id": "新闻id", "title": "标题", "level": "高/中/低", "direction": "利多/利空/中性", "impact": "具体影响"}}
  ]
}}"""
        raw = llm_service._call_llm(
            "你是能源新闻情感分析专家，只输出JSON。",
            prompt,
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = llm_service._parse_llm_response(raw) if raw else None
        if not isinstance(parsed, dict):
            return None

        score = float(np.clip(float(parsed.get("sentiment_score", 0.0)), -1.0, 1.0))
        adjustment = float(np.clip(float(parsed.get("price_adjustment_pct", score * 0.01)), -0.012, 0.012))
        items = parsed.get("news_items") if isinstance(parsed.get("news_items"), list) else []
        if not items:
            items = self.analyze_rule_based(news, current_price).get("news_items", [])

        return {
            "as_of": str(date.today()),
            "sentiment_score": round(score, 4),
            "price_adjustment_pct": round(adjustment, 5),
            "impact_direction": parsed.get("impact_direction", "neutral"),
            "confidence": parsed.get("confidence", "中"),
            "summary": parsed.get("summary", self._summary(score, adjustment)),
            "news_items": items[:8],
            "source": "llm",
            "applied": False,
        }

    def _score_text(self, text: str) -> float:
        value = text.lower()
        score = 0.0
        for pattern, weight in self.BULLISH_PATTERNS.items():
            if pattern.lower() in value:
                score += weight
        for pattern, weight in self.BEARISH_PATTERNS.items():
            if pattern.lower() in value:
                score += weight
        return score

    def _impact_text(self, direction: str, level: str) -> str:
        if direction == "利多":
            return f"{level}强度利多，可能抬升柴油到岸成本与采购锁价压力"
        if direction == "利空":
            return f"{level}强度利空，可能压低柴油成本中枢并延后采购窗口"
        return "暂未形成明确方向，作为背景风险跟踪"

    def _summary(self, sentiment_score: float, adjustment_pct: float) -> str:
        if sentiment_score > 0.05:
            return f"新闻情绪偏利多，模型短端预测上调约 {adjustment_pct * 100:.2f}%"
        if sentiment_score < -0.05:
            return f"新闻情绪偏利空，模型短端预测下调约 {abs(adjustment_pct) * 100:.2f}%"
        return "新闻情绪整体中性，未对价格预测做显著调整"

    def build_adjustment_curve(self, adjustment_pct: float, horizon: int = 30) -> List[float]:
        """Create a bounded, decaying adjustment curve for forecast horizons."""
        pct = float(np.clip(adjustment_pct, -0.012, 0.012))
        return [round(pct * (0.90 ** i), 6) for i in range(max(horizon, 0))]
