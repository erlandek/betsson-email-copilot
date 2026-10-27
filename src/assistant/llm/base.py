"""The LLMClient interface every provider implements.

Agents ask for a Pydantic schema and get back a validated instance plus usage
numbers, whichever provider answers. That keeps agents provider-agnostic and
lets observability record tokens, latency and cost in one place.

Two calls:
- generate(): one prompt in, one schema-validated answer out (triage, summary).
- next_step(): one turn of a tool-calling loop (drafter, scheduler). The model
  either asks for a tool or gives its final answer. The loop itself lives in
  agents/structured.py, so limits and allowlists are enforced in our code, not
  trusted to the model or the provider.
"""

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from assistant.schemas import ToolCall

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel, Generic[T]):
    output: T
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class LLMError(Exception):
    """Provider failure (timeout, rate limit, bad output). Retries and fallback key off this."""


class ToolSpec(BaseModel):
    """What the model is told about a tool. args_model doubles as the JSON schema for its arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: type[BaseModel]


class Message(BaseModel):
    """One entry in a tool loop transcript, in a provider-neutral shape."""

    role: Literal["user", "tool_call", "tool_result"]
    content: str = ""
    tool_call: ToolCall | None = None


class LLMClient(ABC):
    provider: str

    @abstractmethod
    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        """Return an instance of `schema` answering the prompt.

        `system` holds our instructions; `user` holds the data to work on,
        including untrusted email text, which prompts treat as data, never as
        instructions.
        """

    @abstractmethod
    def next_step(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        schema: type[T],
        model: str,
        final: bool = False,
    ) -> ToolCall | LLMResult[T]:
        """Return either the next tool call or the final, schema-validated answer.

        final=True means the call limit is reached and the model must answer now.
        The tools are still described, because the transcript refers to them,
        but the provider is told not to call any.
        """
