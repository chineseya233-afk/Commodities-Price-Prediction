import unittest

from backend.services.news_service import NewsSentimentService


class NewsSentimentTests(unittest.TestCase):
    def test_rule_based_sentiment_maps_bullish_news_to_positive_adjustment(self):
        service = NewsSentimentService(fetch_timeout=0.01)
        news = [
            {
                "title": "OPEC supply cut lifts crude prices",
                "summary": "diesel demand rises while inventories fall",
                "source": "test",
            }
        ]

        result = service.analyze_rule_based(news, current_price=7800.0)

        self.assertGreater(result["sentiment_score"], 0)
        self.assertGreater(result["price_adjustment_pct"], 0)
        self.assertEqual(result["news_items"][0]["level"], "中")

    def test_adjustment_curve_decays_across_horizon(self):
        service = NewsSentimentService()

        curve = service.build_adjustment_curve(0.01, horizon=5)

        self.assertEqual(len(curve), 5)
        self.assertGreater(curve[0], curve[-1])

    def test_neutral_live_news_can_fall_back_to_poc_market_events(self):
        service = NewsSentimentService()
        neutral = service.analyze_rule_based(
            [{"title": "Data centers lift power demand", "summary": "Electricity use grows", "source": "test"}],
            current_price=7800.0,
        )
        fallback = service.analyze_rule_based(service._fallback_news(), current_price=7800.0)

        self.assertEqual(neutral["sentiment_score"], 0.0)
        self.assertNotEqual(fallback["price_adjustment_pct"], 0.0)

    def test_poc_fallback_news_is_chinese_for_frontend_reports(self):
        service = NewsSentimentService()
        fallback = service._fallback_news()

        self.assertGreater(len(fallback), 0)
        self.assertTrue(any("柴油" in item["title"] or "成品油" in item["title"] for item in fallback))
        self.assertTrue(all("remains the key" not in item["title"] for item in fallback))

    def test_parse_china_source_links_filters_relevant_titles(self):
        service = NewsSentimentService()
        html = """
        <html><body>
          <a href="/xw/notice.html">国内成品油价格按机制上调</a>
          <a href="/xw/other.html">无关会议通知</a>
          <a href="https://example.com/diesel.html">柴油库存下降推升市场情绪</a>
        </body></html>
        """

        items = service._parse_html_links(
            html,
            source="测试源",
            base_url="https://www.example.gov.cn/list/",
            keywords=["成品油", "柴油"],
            limit=5,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["title"], "国内成品油价格按机制上调")
        self.assertEqual(items[0]["url"], "https://www.example.gov.cn/xw/notice.html")
        self.assertEqual(items[1]["url"], "https://example.com/diesel.html")


if __name__ == "__main__":
    unittest.main()
