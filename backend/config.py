"""Application configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Global application settings loaded from .env file."""

    # 应用配置
    app_name: str = "CommodityPricePrediction"
    app_env: str = "development"
    debug: bool = True

    # 数据库
    database_url: str = "sqlite+aiosqlite:///./data/commodity_prediction.db"

    # JWT 和会话
    jwt_secret_key: str = "poc-secret-key-change-in-production-2026"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440
    session_idle_minutes: int = 30
    remember_me_days: int = 14
    session_cookie_name: str = "commodity_session"
    cookie_secure: bool = False

    # POC 演示用户。非本地使用时请在 .env 中覆盖。
    admin_default_password: str = "Admin123456@"
    executive_default_password: str = "Exec123456@"
    procurement_default_password: str = "Proc123456@"

    # 通用 OpenAI-compatible LLM 端点。DeepSeek 是默认示例，
    # 也可以通过这些字段配置其他兼容供应商。
    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = ""
    openai_compatible_provider: str = ""

    # 向后兼容的旧变量。新环境文件优先使用 OPENAI_COMPATIBLE_* 或 LLM_*。
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-v4-pro"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"

    # EIA
    eia_api_key: str = "demo"
    eia_base_url: str = "https://api.eia.gov/v2"
    eia_diesel_series: str = "EER_EPD2DXL0_PF4_Y35NY_DPG"

    # FRED
    fred_api_key: str = ""
    fred_base_url: str = "https://api.stlouisfed.org/fred"

    # 模型
    model_dir: str = "backend/ml/trained_models"
    prediction_horizon: int = 30
    lookback_window: int = 30
    data_refresh_seconds: int = 300
    market_data_start_date: str = "2006-06-01"

    # 临时目录
    temp_dir: str = "data/temp"
    trained_models_dir: str = "backend/ml/trained_models"

    # 服务端口。当前前端由 FastAPI 同端口托管，不再单独配置 3000 端口。
    backend_port: int = 8000

    @property
    def resolved_llm_api_key(self) -> str:
        """Resolve AI key with new generic variables first, then legacy aliases."""
        return self.openai_compatible_api_key or self.llm_api_key or self.deepseek_api_key

    @property
    def resolved_llm_base_url(self) -> str:
        """Resolve the OpenAI-compatible base URL."""
        return (
            self.openai_compatible_base_url
            or self.llm_base_url
            or self.deepseek_base_url
            or "https://api.deepseek.com/v1"
        )

    @property
    def resolved_llm_model(self) -> str:
        """Resolve the OpenAI-compatible model name."""
        return (
            self.openai_compatible_model
            or self.llm_model
            or self.deepseek_model
            or "deepseek-v4-pro"
        )

    @property
    def resolved_llm_provider(self) -> str:
        """Resolve a display-only provider label."""
        if self.openai_compatible_provider:
            return self.openai_compatible_provider
        base_url = self.resolved_llm_base_url.lower()
        if "deepseek" in base_url:
            return "deepseek"
        if "openai" in base_url:
            return "openai"
        return "openai-compatible"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
