from assistant.config import Provider, Settings
from assistant.llm.base import LLMClient, LLMError, LLMResult
from assistant.llm.fake import FakeLLM


def get_llm_client(settings: Settings, provider: Provider | None = None) -> LLMClient:
    """Build the client for the configured provider (fake by default)."""
    provider = provider or settings.llm_provider
    if provider == "fake":
        return FakeLLM()
    # Imported lazily so fake mode never needs the provider SDKs installed.
    if provider == "openai":
        from assistant.llm.openai_provider import OpenAIClient

        key = settings.openai_api_key
        return OpenAIClient(key.get_secret_value() if key else None, **_limits(settings))
    if provider == "anthropic":
        from assistant.llm.anthropic_provider import AnthropicClient

        key = settings.anthropic_api_key
        return AnthropicClient(key.get_secret_value() if key else None, **_limits(settings))
    raise ValueError(f"unknown provider: {provider}")


def _limits(settings: Settings) -> dict:
    return {"max_retries": settings.llm_max_retries, "timeout": settings.llm_timeout_seconds}


__all__ = ["FakeLLM", "LLMClient", "LLMError", "LLMResult", "get_llm_client"]
