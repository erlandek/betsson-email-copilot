"""Real providers, tested against stand-in SDK clients (no keys, no network).

The stand-ins return objects shaped like the SDK responses and record each
request, so we can check what we send: schemas, tools, tool_choice, and that a
tool call's id comes back with its result.
"""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import openai
import pytest

from assistant.llm import LLMError
from assistant.llm.anthropic_provider import AnthropicClient
from assistant.llm.base import Message, ToolSpec
from assistant.llm.openai_provider import OpenAIClient
from assistant.schemas import DraftReply, ToolCall
from assistant.tools import TOOLS

DRAFT_JSON = '{"to": ["sofia@example.com"], "subject": "Re: Q4", "body": "Hi Sofia,\\n\\nBest regards, Erland"}'
SPECS: list[ToolSpec] = [TOOLS["lookup_contact"].spec]
REQUEST = httpx2.Request("POST", "https://example.invalid")


class Recorder:
    """Plays back canned responses and keeps every request it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, **request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# --- Anthropic ---------------------------------------------------------------


def claude_text(text, stop="end_turn"):
    return NS(
        stop_reason=stop,
        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
        usage=NS(input_tokens=1000, output_tokens=200),
    )


def claude_tool(call_id="toolu_1"):
    block = NS(type="tool_use", id=call_id, name="lookup_contact", input={"email": "sofia@example.com"})
    content = [NS(type="thinking", thinking=""), block]
    return NS(stop_reason="tool_use", content=content, usage=NS(input_tokens=1, output_tokens=1))


def claude(*responses):
    recorder = Recorder(*responses)
    return AnthropicClient("key", client=NS(messages=NS(create=recorder))), recorder


def test_anthropic_generate_returns_validated_output_with_usage_and_cost():
    client, rec = claude(claude_text(DRAFT_JSON))
    result = client.generate("Draft.", "email text", DraftReply, "claude-sonnet-5-5")
    assert result.output.subject == "Re: Q4"
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    assert result.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)
    request = rec.requests[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in request and "temperature" not in request


def test_anthropic_tool_loop_round_trip():
    client, rec = claude(claude_tool(), claude_text(DRAFT_JSON))
    step = client.next_step("Draft.", [Message(role="user", content="email")], SPECS, DraftReply, "m")
    assert step == ToolCall(id="toolu_1", name="lookup_contact", args={"email": "sofia@example.com"})
    first = rec.requests[0]
    assert first["tools"][0]["strict"] is True and first["tools"][0]["input_schema"]["additionalProperties"] is False
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    transcript = [
        Message(role="user", content="email"),
        Message(role="tool_call", tool_call=step),
        Message(role="tool_result", content="Contact sofia: VIP", tool_call=step),
    ]
    final = client.next_step("Draft.", transcript, SPECS, DraftReply, "m", final=True)
    assert final.output.subject == "Re: Q4"
    second = rec.requests[1]
    assert second["tool_choice"] == {"type": "none"}
    _, assistant, result = second["messages"]
    assert assistant["content"][0].type == "thinking"  # replayed unchanged, thinking included
    assert result["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Contact sofia: VIP"}


def test_anthropic_invalid_json_raises_validation_error():
    client, _ = claude(claude_text('{"subject": "x"}'))
    with pytest.raises(ValueError):  # pydantic.ValidationError, which the agents retry once
        client.generate("s", "u", DraftReply, "m")


@pytest.mark.parametrize(
    "response",
    [
        claude_text("", stop="refusal"),
        claude_text("{", stop="max_tokens"),
        anthropic.APIConnectionError(request=REQUEST),
    ],
)
def test_anthropic_failures_become_llm_errors(response):
    client, _ = claude(response)
    with pytest.raises(LLMError):
        client.generate("s", "u", DraftReply, "m")


# --- OpenAI ------------------------------------------------------------------


class Item(NS):
    def model_dump(self, exclude_none=False):
        return dict(vars(self))


def gpt_text(text, status="completed"):
    return NS(
        status=status,
        incomplete_details=None,
        output=[Item(type="message", content=[])],
        output_text=text,
        usage=NS(input_tokens=1000, output_tokens=200),
    )


def gpt_tool(call_id="call_1"):
    items = [
        Item(type="reasoning", encrypted_content="opaque"),
        Item(type="function_call", call_id=call_id, name="lookup_contact", arguments='{"email": "sofia@example.com"}'),
    ]
    return NS(status="completed", output=items, output_text="", usage=NS(input_tokens=1, output_tokens=1))


def gpt(*responses):
    recorder = Recorder(*responses)
    return OpenAIClient("key", client=NS(responses=NS(create=recorder))), recorder


def test_openai_generate_returns_validated_output_with_usage_and_cost():
    client, rec = gpt(gpt_text(DRAFT_JSON))
    result = client.generate("Draft.", "email text", DraftReply, "gpt-5.4")
    assert result.output.subject == "Re: Q4"
    assert result.cost_usd == pytest.approx((1000 * 2.5 + 200 * 15) / 1e6)
    request = rec.requests[0]
    assert request["text"]["format"]["strict"] is True and request["store"] is False
    assert request["instructions"] == "Draft."


def test_openai_tool_loop_round_trip():
    client, rec = gpt(gpt_tool(), gpt_text(DRAFT_JSON))
    step = client.next_step("Draft.", [Message(role="user", content="email")], SPECS, DraftReply, "m")
    assert step == ToolCall(id="call_1", name="lookup_contact", args={"email": "sofia@example.com"})
    first = rec.requests[0]
    assert first["tools"][0]["type"] == "function" and first["tools"][0]["strict"] is True
    assert first["tool_choice"] == "auto" and first["parallel_tool_calls"] is False

    transcript = [
        Message(role="user", content="email"),
        Message(role="tool_call", tool_call=step),
        Message(role="tool_result", content="Contact sofia: VIP", tool_call=step),
    ]
    client.next_step("Draft.", transcript, SPECS, DraftReply, "m", final=True)
    second = rec.requests[1]
    assert second["tool_choice"] == "none"
    types = [item.get("type", "message") for item in second["input"]]
    assert types == ["message", "reasoning", "function_call", "function_call_output"]
    assert second["input"][-1] == {"type": "function_call_output", "call_id": "call_1", "output": "Contact sofia: VIP"}


@pytest.mark.parametrize(
    "response", [gpt_text("", status="incomplete"), gpt_text(""), openai.APIConnectionError(request=REQUEST)]
)
def test_openai_failures_become_llm_errors(response):
    client, _ = gpt(response)
    with pytest.raises(LLMError):
        client.generate("s", "u", DraftReply, "m")
