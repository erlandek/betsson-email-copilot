"""Anthropic provider, on the Messages API (anthropic SDK 2.1).

- Structured output: output_config.format with a JSON schema made from the
  Pydantic model; we still validate the reply with Pydantic ourselves.
- Tools: each ToolSpec becomes a strict tool whose input_schema comes from its
  args_model. One tool call per turn, because our loop runs them one at a time.
- Current models reject a forced tool_choice, so we never force a call:
  "auto" while tools are allowed, "none" once the limit is reached.
- The assistant turn that asked for a tool is replayed unchanged on the next
  request, because it may hold thinking blocks that must be sent back as they came.
"""

import time
from typing import Any

import anthropic
from anthropic import transform_schema

from assistant.llm.base import LLMClient, LLMError, LLMResult, Message, T, ToolSpec
from assistant.llm.pricing import cost_usd
from assistant.schemas import ToolCall

MAX_TOKENS = 16000  # room for thinking plus a short structured reply
EFFORT = "low"  # triage and drafting are short tasks; low effort keeps latency and cost down
# Failures worth our one retry. Anything else (bad key, bad request) is a setup bug and stops the run.
TRANSIENT = (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str | None, client: Any = None):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set; add it to .env or use LLM_PROVIDER=fake")
        self._client = client or anthropic.Anthropic(api_key=api_key)
        self._turns: dict[str, list] = {}  # tool call id -> the assistant content that made the call

    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        return self.next_step(system, [Message(role="user", content=user)], [], schema, model, final=True)

    def next_step(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        schema: type[T],
        model: str,
        final: bool = False,
    ) -> ToolCall | LLMResult[T]:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": self._to_messages(messages),
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": transform_schema(schema)}},
        }
        if tools:
            request["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": transform_schema(t.args_model),
                    "strict": True,
                }
                for t in tools
            ]
            request["tool_choice"] = {"type": "none"} if final else {"type": "auto", "disable_parallel_tool_use": True}

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**request)
        except TRANSIENT as error:
            raise LLMError(f"anthropic: {type(error).__name__}: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            raise LLMError("anthropic: the model declined to answer")
        if response.stop_reason == "max_tokens":
            raise LLMError("anthropic: answer was cut off at max_tokens")

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is not None:
            self._turns[tool_use.id] = response.content
            return ToolCall(id=tool_use.id, name=tool_use.name, args=dict(tool_use.input))

        text = next((b.text for b in response.content if b.type == "text"), "")
        output = schema.model_validate_json(text)  # a ValidationError here triggers our one retry
        usage = response.usage
        return LLMResult[schema](
            output=output,
            provider=self.provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
        )

    def _to_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Map our neutral transcript to Anthropic turns, merging neighbours that share a role."""
        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                role, blocks = "user", [{"type": "text", "text": m.content}]
            elif m.role == "tool_call":
                call = m.tool_call
                role = "assistant"
                blocks = list(self._turns.get(call.id, [])) or [
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
                ]
            else:
                role = "user"
                blocks = [{"type": "tool_result", "tool_use_id": m.tool_call.id, "content": m.content}]
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(blocks)
            else:
                turns.append({"role": role, "content": blocks})
        return turns
