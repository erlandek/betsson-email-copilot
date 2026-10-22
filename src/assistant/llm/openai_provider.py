"""OpenAI provider, on the Responses API (openai SDK 3.36).

- Structured output: text.format json_schema in strict mode, built from the
  Pydantic model with the SDK's own strict-schema helper; we still validate
  the reply with Pydantic ourselves.
- Tools: each ToolSpec becomes a strict function tool. parallel_tool_calls is
  off because our loop runs one call at a time.
- tool_choice is "auto" while tools are allowed and "none" once the limit is hit.
- store=False so email content is not kept on OpenAI's side. Reasoning items
  then come back encrypted, and we replay the model's output items unchanged.
"""

import json
import time
from typing import Any

import openai
from openai.lib._pydantic import to_strict_json_schema  # what responses.parse uses internally

from assistant.llm.base import LLMClient, LLMError, LLMResult, Message, T, ToolSpec
from assistant.llm.pricing import cost_usd
from assistant.schemas import ToolCall

MAX_OUTPUT_TOKENS = 16000
# Still failing after the SDK's own retries: these become LLMError, which moves the call to the
# fallback provider. Anything else (bad key, bad request) is a setup bug and stops the run.
TRANSIENT = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str | None, client: Any = None, max_retries: int = 2, timeout: float = 60.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set; add it to .env or use LLM_PROVIDER=fake")
        # The SDK retries connection errors, 429 and 5xx with exponential backoff, max_retries times.
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._turns: dict[str, list] = {}  # tool call id -> the output items that made the call

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
            "instructions": system,
            "input": self._to_input(messages),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": to_strict_json_schema(schema),
                    "strict": True,
                }
            },
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": to_strict_json_schema(t.args_model),
                    "strict": True,
                }
                for t in tools
            ]
            request["tool_choice"] = "none" if final else "auto"
            request["parallel_tool_calls"] = False

        started = time.perf_counter()
        try:
            response = self._client.responses.create(**request)
        except TRANSIENT as error:
            raise LLMError(f"openai: {type(error).__name__}: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status == "incomplete":
            raise LLMError(f"openai: incomplete answer ({response.incomplete_details})")

        call = next((item for item in response.output if item.type == "function_call"), None)
        if call is not None:
            self._turns[call.call_id] = [item.model_dump(exclude_none=True) for item in response.output]
            return ToolCall(id=call.call_id, name=call.name, args=json.loads(call.arguments or "{}"))

        if not response.output_text:
            raise LLMError("openai: no answer text (the model may have refused)")
        output = schema.model_validate_json(response.output_text)  # a ValidationError triggers our one retry
        usage = response.usage
        input_tokens, output_tokens = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        return LLMResult[schema](
            output=output,
            provider=self.provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd(model, input_tokens, output_tokens),
        )

    def _to_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Map our neutral transcript to Responses API input items."""
        items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                items.append({"role": "user", "content": m.content})
            elif m.role == "tool_call":
                call = m.tool_call
                args = json.dumps(call.args)
                rebuilt = {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": args}
                items.extend(self._turns.get(call.id) or [rebuilt])
            else:
                items.append({"type": "function_call_output", "call_id": m.tool_call.id, "output": m.content})
        return items
