"""Fall back from one provider to the other when the first one fails.

Retries with backoff (429, 5xx, timeouts) already happen inside each SDK. Only
when those are used up does a provider raise LLMError, and then the same call
goes to the other provider, with that provider's model for the same task.
Validation errors are not a reason to switch: the agents retry those once and
then send the email to human review.
"""

from assistant.llm.base import LLMClient, LLMError, LLMResult, Message, T, ToolSpec
from assistant.observability import Observability
from assistant.schemas import ToolCall


class FallbackLLM(LLMClient):
    def __init__(self, primary: LLMClient, secondary: LLMClient, model_map: dict[str, str], obs: Observability):
        self.primary, self.secondary, self.model_map, self.obs = primary, secondary, model_map, obs
        self.provider = primary.provider

    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        try:
            return self.primary.generate(system, user, schema, model)
        except LLMError as error:
            self._note(error, model)
            return self.secondary.generate(system, user, schema, self.model_map.get(model, model))

    def next_step(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        schema: type[T],
        model: str,
        final: bool = False,
    ) -> ToolCall | LLMResult[T]:
        try:
            return self.primary.next_step(system, messages, tools, schema, model, final=final)
        except LLMError as error:
            self._note(error, model)
            fallback_model = self.model_map.get(model, model)
            return self.secondary.next_step(system, messages, tools, schema, fallback_model, final=final)

    def _note(self, error: LLMError, model: str) -> None:
        self.obs.metrics.inc("llm_fallbacks_total", primary=self.primary.provider, secondary=self.secondary.provider)
        self.obs.log.warning("llm_fallback", primary=self.primary.provider, secondary=self.secondary.provider,
                             model=model, error=str(error))
