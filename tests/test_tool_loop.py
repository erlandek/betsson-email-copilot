import pytest

from assistant.agents.structured import AgentError, call_structured, execute_tool, run_tool_loop
from assistant.llm.base import LLMClient, LLMError, LLMResult
from assistant.schemas import DraftReply, ToolCall, TriageResult

DRAFT = DraftReply(to=["a@example.com"], subject="Re: x", body="Hi")


class ScriptedLLM(LLMClient):
    """Test double: plays back a list of steps and records what it was offered."""

    provider = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)
        self.offered = []

    def generate(self, system, user, schema, model):
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResult[schema](output=step, provider=self.provider, model=model)

    def next_step(self, system, messages, tools, schema, model):
        self.offered.append([t.name for t in tools])
        step = self.steps.pop(0) if self.steps else DRAFT
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ToolCall):
            return step
        return LLMResult[schema](output=step, provider=self.provider, model=model)


def test_execute_tool_enforces_allowlist(deps):
    result = execute_tool(deps, "scheduler", ToolCall(name="search_knowledge", args={"query": "sandbox"}))
    assert result.error and "not allowed" in result.error


def test_execute_tool_validates_arguments(deps):
    bad = execute_tool(deps, "scheduler", ToolCall(name="check_availability",
                                                   args={"start": "2026-10-27T10:00:00+01:00", "end": "yesterday"}))
    assert bad.error and "invalid arguments" in bad.error
    reversed_window = execute_tool(deps, "scheduler", ToolCall(name="check_availability", args={
        "start": "2026-10-28T10:00:00+01:00", "end": "2026-10-27T10:00:00+01:00"}))
    assert reversed_window.error
    extra = execute_tool(deps, "drafter", ToolCall(name="get_preferences", args={"drop_tables": True}))
    assert extra.error


def test_execute_tool_returns_results(deps):
    result = execute_tool(deps, "drafter", ToolCall(name="search_knowledge", args={"query": "sandbox credentials"}))
    assert result.error is None and "[F02]" in result.result
    contact = execute_tool(deps, "drafter", ToolCall(name="lookup_contact", args={"email": "sofia.lind@example.com"}))
    assert "VIP" in contact.result


def test_tool_loop_stops_offering_tools_after_the_limit(deps):
    call = ToolCall(name="get_preferences", args={})
    deps.llm = ScriptedLLM([call] * 4 + [DRAFT])
    output, results = run_tool_loop(deps, "drafter", "s", "u", DraftReply, "m")
    assert output.output == DRAFT
    assert len(results) == 4
    assert deps.llm.offered[-1] == []  # fifth step: no tools offered, so the model must answer


def test_tool_loop_raises_if_model_ignores_the_limit(deps):
    deps.llm = ScriptedLLM([ToolCall(name="get_preferences", args={})] * 5)
    with pytest.raises(AgentError):
        run_tool_loop(deps, "drafter", "s", "u", DraftReply, "m")


def test_denied_calls_count_towards_the_limit(deps):
    deps.llm = ScriptedLLM([ToolCall(name="send_email", args={})] * 4 + [DRAFT])
    _, results = run_tool_loop(deps, "drafter", "s", "u", DraftReply, "m")
    assert len(results) == 4 and all(r.error for r in results)


def test_tool_loop_retries_once_then_escalates(deps):
    deps.llm = ScriptedLLM([LLMError("bad json"), DRAFT])
    output, _ = run_tool_loop(deps, "drafter", "s", "u", DraftReply, "m")
    assert output.output == DRAFT
    deps.llm = ScriptedLLM([LLMError("bad json"), LLMError("still bad")])
    with pytest.raises(AgentError):
        run_tool_loop(deps, "drafter", "s", "u", DraftReply, "m")


def test_call_structured_retries_once_then_escalates(deps):
    good = TriageResult(intent="fyi", priority="low", confidence=0.9, rationale="r")
    deps.llm = ScriptedLLM([LLMError("bad"), good])
    assert call_structured(deps, "s", "u", TriageResult, "m").output == good
    deps.llm = ScriptedLLM([LLMError("bad"), LLMError("bad again")])
    with pytest.raises(AgentError):
        call_structured(deps, "s", "u", TriageResult, "m")


def test_fake_drafter_follows_its_script(deps):
    from assistant.agents.drafter import draft_reply
    from assistant.agents.summariser import summarise_thread

    email = deps.mail.get_email("e04")
    summary = summarise_thread(deps, deps.mail.get_thread(email.thread_id))
    draft, results = draft_reply(deps, email, summary)
    assert [r.name for r in results] == ["lookup_contact", "search_knowledge", "get_preferences"]
    assert draft.to == ["mark.jensen@nordwind.example"]
    assert set(draft.facts_used) <= {"F02", "F03", "F04"} and draft.facts_used
    assert draft.body.endswith("Best regards, Erland")  # sign-off from seeded preferences
