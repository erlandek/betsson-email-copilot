import pytest
from pydantic import ValidationError

from assistant.config import Settings


def test_defaults_need_no_keys(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "fake"
    assert settings.approval_mode == "all"
    assert settings.confidence_threshold == 0.6


def test_env_overrides_and_model_lookup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_TRIAGE_MODEL", "small-model")
    settings = Settings(_env_file=None)
    assert settings.model_for("triage") == "small-model"
    assert settings.model_for("draft", provider="openai") == settings.openai_draft_model
    assert settings.model_for("triage", provider="fake") == "fake-triage"


def test_invalid_values_fail_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "opneai")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
