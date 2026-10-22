"""The LLMClient interface every provider implements.

Agents ask for a Pydantic schema and get back a validated instance plus usage
numbers, whichever provider answers. That keeps agents provider-agnostic and
lets observability record tokens, latency and cost in one place.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

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


class LLMClient(ABC):
    provider: str

    @abstractmethod
    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        """Return an instance of `schema` answering the prompt.

        `system` holds our instructions; `user` holds the data to work on,
        including untrusted email text, which prompts treat as data, never as
        instructions.
        """
