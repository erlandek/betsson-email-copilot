"""Wraps any LLMClient with a span, a log line and metrics for every call.

Span names and attributes follow the OpenTelemetry GenAI conventions
("chat <model>", gen_ai.provider.name, gen_ai.usage.input_tokens, ...), so a
standard backend can read them. Prompts and email text are not recorded:
they are personal data, and the tokens, latency and outcome are what matter.
"""

import time

from opentelemetry.trace import Status, StatusCode

from assistant.llm.base import LLMClient, LLMResult, Message, T, ToolSpec
from assistant.observability import Observability
from assistant.schemas import ToolCall


class InstrumentedLLM(LLMClient):
    def __init__(self, inner: LLMClient, obs: Observability):
        self.inner, self.obs = inner, obs
        self.provider = inner.provider

    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        return self._traced(model, schema, lambda: self.inner.generate(system, user, schema, model))

    def next_step(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        schema: type[T],
        model: str,
        final: bool = False,
    ) -> ToolCall | LLMResult[T]:
        return self._traced(
            model, schema, lambda: self.inner.next_step(system, messages, tools, schema, model, final=final)
        )

    def _traced(self, model: str, schema: type, call):
        obs, provider = self.obs, self.provider
        with obs.tracer.start_as_current_span(f"chat {model}", record_exception=False) as span:
            span.set_attributes({
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
                "assistant.schema": schema.__name__,
            })
            started = time.perf_counter()
            try:
                result = call()
            except Exception as error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                obs.metrics.inc("llm_errors_total", provider=provider, error=type(error).__name__)
                obs.log.warning("llm_call_failed", provider=provider, model=model, error=str(error))
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            obs.metrics.observe("llm_latency_ms", latency_ms, provider=provider)
            obs.metrics.inc("llm_calls_total", provider=provider, model=model)

            if isinstance(result, ToolCall):
                span.set_attribute("assistant.tool_requested", result.name)
                obs.log.info("llm_call", provider=provider, model=model, latency_ms=round(latency_ms, 1),
                             tool_requested=result.name)
                return result

            span.set_attributes({
                "gen_ai.usage.input_tokens": result.input_tokens,
                "gen_ai.usage.output_tokens": result.output_tokens,
                "assistant.cost_usd": result.cost_usd,
            })
            obs.metrics.inc("llm_tokens_total", result.input_tokens, direction="input")
            obs.metrics.inc("llm_tokens_total", result.output_tokens, direction="output")
            obs.metrics.inc("llm_cost_usd_total", result.cost_usd)
            obs.log.info("llm_call", provider=provider, model=model, schema=schema.__name__,
                         latency_ms=round(latency_ms, 1), input_tokens=result.input_tokens,
                         output_tokens=result.output_tokens, cost_usd=round(result.cost_usd, 6))
            return result
