"""OpenAI provider (stub).

Implemented in a later phase, after checking the installed SDK version and the
current docs for structured outputs. For now it validates config so a missing
key fails at startup rather than mid-run.
"""

from assistant.llm.base import LLMClient, LLMResult, Message, T, ToolSpec
from assistant.schemas import ToolCall


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set; add it to .env or use LLM_PROVIDER=fake")
        self._api_key = api_key

    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        raise NotImplementedError("OpenAI provider is implemented in a later phase")

    def next_step(
        self, system: str, messages: list[Message], tools: list[ToolSpec], schema: type[T], model: str
    ) -> ToolCall | LLMResult[T]:
        # Will map ToolSpec.args_model to the provider's native tool definitions.
        raise NotImplementedError("OpenAI provider is implemented in a later phase")
