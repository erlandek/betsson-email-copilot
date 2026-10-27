"""Logs, traces, metrics, retries and provider fallback (DECISIONS.md, D36 and D37)."""

import json

import pytest
from pydantic import ValidationError
from test_tool_loop import DRAFT, ScriptedLLM

from assistant.agents.structured import call_structured
from assistant.config import Settings
from assistant.deps import build_llm
from assistant.graph import build_graph, process_email
from assistant.llm import FakeLLM, LLMError
from assistant.llm.fallback import FallbackLLM
from assistant.llm.instrumented import InstrumentedLLM
from assistant.llm.openai_provider import OpenAIClient
from assistant.observability.waterfall import find_trace, load_spans, render
from assistant.schemas import DraftReply, HumanDecision


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spans_of(deps):
    deps.obs.close()
    return load_spans(deps.settings.logs_dir / "traces.jsonl")


def test_one_trace_per_run_with_nested_node_llm_and_tool_spans(deps):
    process_email(build_graph(), deps, "e04")
    spans = spans_of(deps)
    assert len({s["context"]["trace_id"] for s in spans}) == 1
    by_id = {s["context"]["span_id"]: s for s in spans}
    parent = {s["name"]: by_id[s["parent_id"]]["name"] for s in spans if s["parent_id"]}
    assert parent["node triage"] == "email_run"
    assert parent["chat fake-triage"] == "node triage"
    assert parent["execute_tool search_knowledge"] == "node draft"
    chat = next(s for s in spans if s["name"] == "chat fake-triage")
    assert chat["attributes"]["gen_ai.provider.name"] == "fake"
    assert chat["attributes"]["gen_ai.usage.input_tokens"] > 0


def test_waiting_for_approval_is_not_an_error(deps):
    process_email(build_graph(), deps, "e04")
    approval = next(s for s in spans_of(deps) if s["name"] == "node human_approval")
    assert approval["status"]["status_code"] != "ERROR"
    assert approval["attributes"]["assistant.waiting_for_human"] is True


def test_traces_hold_no_email_text(deps):
    process_email(build_graph(), deps, "e04")
    deps.obs.close()
    body = deps.mail.get_email("e04").body
    assert body[:40] not in (deps.settings.logs_dir / "traces.jsonl").read_text(encoding="utf-8")


def test_log_lines_carry_run_and_trace_ids(deps):
    process_email(build_graph(), deps, "e10")
    spans = spans_of(deps)
    lines = read_jsonl(deps.settings.logs_dir / "app.jsonl")
    run_line = next(line for line in lines if line["event"] == "email_run")
    assert run_line["email_id"] == "e10" and run_line["run_id"]
    assert "0x" + run_line["trace_id"] == spans[-1]["context"]["trace_id"]


def test_handled_failure_is_logged_counted_and_marked_on_the_span(deps):
    class BrokenDrafts(FakeLLM):
        def next_step(self, *args, **kwargs):
            raise LLMError("provider down")

    deps.llm = InstrumentedLLM(BrokenDrafts(), deps.obs)
    process_email(build_graph(), deps, "e04")
    errors = read_jsonl(deps.settings.logs_dir / "errors.jsonl")
    assert errors and errors[0]["node"] == "draft" and errors[0]["trace_id"]
    assert deps.obs.metrics.counters["errors_total{node=draft}"] == 1
    draft = next(s for s in spans_of(deps) if s["name"] == "node draft")
    assert draft["status"]["status_code"] == "ERROR"


def test_validation_failures_are_counted(deps):
    bad = ValidationError.from_exception_data("DraftReply", [])
    deps.llm = ScriptedLLM([bad, DRAFT])
    call_structured(deps, "s", "u", DraftReply, "m")
    assert deps.obs.metrics.counters["validation_failures_total{schema=DraftReply}"] == 1


def test_metrics_and_decisions_are_saved_on_close(deps):
    process_email(build_graph(), deps, "e04", decide=lambda view: HumanDecision(decision="approve"))
    deps.obs.close()
    saved = json.loads((deps.settings.logs_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["counters"]["decisions_total{decision=approve}"] == 1
    assert saved["counters"]["email_runs_total{route=reply,status=done}"] == 1
    assert saved["histograms"]["node_latency_ms{node=draft}"]["count"] == 1
    assert "llm_calls_total" in deps.obs.metrics.summary()


def test_waterfall_shows_the_run(deps):
    process_email(build_graph(), deps, "e07")
    spans = spans_of(deps)
    text = render(spans, find_trace(spans, "e07"))
    assert "email e07" in text and "execute_tool check_availability" in text and "waiting for human" in text


# --- Retries and fallback ----------------------------------------------------


class Down(ScriptedLLM):
    provider = "down"

    def __init__(self, error):
        super().__init__([])
        self.error = error

    def generate(self, *args, **kwargs):
        raise self.error

    def next_step(self, *args, **kwargs):
        raise self.error


def test_fallback_uses_the_other_provider_and_its_model(deps):
    backup = ScriptedLLM([DRAFT])
    llm = FallbackLLM(Down(LLMError("timeout")), backup, {"gpt-5.4": "claude-sonnet-5-5"}, deps.obs)
    result = llm.generate("s", "u", DraftReply, "gpt-5.4")
    assert result.model == "claude-sonnet-5-5" and result.provider == "scripted"
    assert deps.obs.metrics.counters["llm_fallbacks_total{primary=down,secondary=scripted}"] == 1


def test_fallback_in_the_tool_loop(deps):
    llm = FallbackLLM(Down(LLMError("rate limited")), ScriptedLLM([DRAFT]), {}, deps.obs)
    assert llm.next_step("s", [], [], DraftReply, "m").output == DRAFT


def test_invalid_output_does_not_switch_provider(deps):
    bad = ValidationError.from_exception_data("DraftReply", [])
    backup = ScriptedLLM([DRAFT])
    llm = FallbackLLM(Down(bad), backup, {}, deps.obs)
    with pytest.raises(ValidationError):
        llm.generate("s", "u", DraftReply, "m")
    assert backup.steps == [DRAFT]  # untouched: the agent's own retry handles bad output


def settings(**keys):
    return Settings(_env_file=None, **keys)


def test_fallback_is_wired_only_when_the_other_key_is_set(deps):
    assert isinstance(build_llm(settings(), deps.obs), InstrumentedLLM)  # fake: no fallback
    only_openai = settings(llm_provider="openai", openai_api_key="k")
    assert isinstance(build_llm(only_openai, deps.obs), InstrumentedLLM)
    both = settings(llm_provider="openai", openai_api_key="k", anthropic_api_key="k")
    llm = build_llm(both, deps.obs)
    assert isinstance(llm, FallbackLLM)
    assert llm.model_map == {"gpt-6-luna": "claude-haiku-5-5", "gpt-6.1-terra": "claude-sonnet-5-5"}
    assert not isinstance(build_llm(both.model_copy(update={"llm_fallback": False}), deps.obs), FallbackLLM)


def test_sdk_retries_and_timeout_come_from_settings():
    client = OpenAIClient("k", max_retries=5, timeout=12.0)._client
    assert client.max_retries == 5 and client.timeout == 12.0
