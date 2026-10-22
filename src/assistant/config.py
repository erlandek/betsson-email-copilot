"""Application settings, read from environment variables and an optional .env file.

Pydantic validates every value at startup, so a typo such as LLM_PROVIDER=opneai
fails immediately with a clear error instead of halfway through a run.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai", "anthropic"]
# "triage" covers classification; "draft" covers summaries, replies and scheduling.
ModelTask = Literal["triage", "draft"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "fake"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # A small, fast model for triage and a stronger one for drafting.
    openai_triage_model: str = "gpt-5.4-mini"
    openai_draft_model: str = "gpt-5.4"
    anthropic_triage_model: str = "claude-haiku-5-5"
    anthropic_draft_model: str = "claude-sonnet-5-5"

    # "all": every outgoing action needs approval, as the brief requires.
    # "risk_based" is documented as the production option but not the default.
    # Reliability. The SDKs retry 429, 5xx and timeouts with exponential backoff
    # llm_max_retries times; after that the call falls back to the other real
    # provider, if llm_fallback is on and that provider's key is set.
    llm_max_retries: int = Field(default=2, ge=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_fallback: bool = True

    approval_mode: Literal["all", "risk_based"] = "all"
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_tool_calls: int = Field(default=4, ge=1)
    user_email: str = "erland@example.com"  # whose mailbox this is
    timezone: str = "Europe/Stockholm"  # the user's time zone, for working hours and scheduling
    # chroma: semantic search with a local embedding model; keyword: word overlap, no download.
    knowledge_backend: Literal["chroma", "keyword"] = "chroma"

    data_dir: Path = Path("data")
    runtime_dir: Path = Path("runtime")  # mock mailbox state (sent.json, calendar.json)
    logs_dir: Path = Path("logs")
    otel_exporter_otlp_endpoint: str | None = None

    def model_for(self, task: ModelTask, provider: Provider | None = None) -> str:
        """Return the configured model name for a task on a provider."""
        provider = provider or self.llm_provider
        if provider == "fake":
            return f"fake-{task}"
        return getattr(self, f"{provider}_{task}_model")


def get_settings() -> Settings:
    return Settings()
