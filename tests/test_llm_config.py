import unittest

from backend.config import Settings


class LLMConfigTests(unittest.TestCase):
    def test_openai_compatible_settings_take_priority(self):
        settings = Settings(
            _env_file=None,
            openai_compatible_api_key="new-key",
            openai_compatible_base_url="https://example.ai/v1",
            openai_compatible_model="example-model",
            openai_compatible_provider="example",
            llm_api_key="old-key",
            llm_base_url="https://old.example/v1",
            llm_model="old-model",
            deepseek_api_key="deepseek-key",
        )

        self.assertEqual(settings.resolved_llm_api_key, "new-key")
        self.assertEqual(settings.resolved_llm_base_url, "https://example.ai/v1")
        self.assertEqual(settings.resolved_llm_model, "example-model")
        self.assertEqual(settings.resolved_llm_provider, "example")

    def test_llm_aliases_are_supported(self):
        settings = Settings(
            _env_file=None,
            llm_api_key="llm-key",
            llm_base_url="https://provider.example/v1",
            llm_model="provider-model",
        )

        self.assertEqual(settings.resolved_llm_api_key, "llm-key")
        self.assertEqual(settings.resolved_llm_base_url, "https://provider.example/v1")
        self.assertEqual(settings.resolved_llm_model, "provider-model")
        self.assertEqual(settings.resolved_llm_provider, "openai-compatible")

    def test_deepseek_legacy_settings_remain_supported(self):
        settings = Settings(
            _env_file=None,
            llm_api_key="",
            llm_base_url="",
            llm_model="",
            deepseek_api_key="deepseek-key",
            deepseek_base_url="https://api.deepseek.com/v1",
            deepseek_model="deepseek-chat",
        )

        self.assertEqual(settings.resolved_llm_api_key, "deepseek-key")
        self.assertEqual(settings.resolved_llm_base_url, "https://api.deepseek.com/v1")
        self.assertEqual(settings.resolved_llm_model, "deepseek-chat")
        self.assertEqual(settings.resolved_llm_provider, "deepseek")


if __name__ == "__main__":
    unittest.main()
