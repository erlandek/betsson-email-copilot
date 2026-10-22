"""Schema-validated LLM calls with one retry, and a bounded tool-calling loop.

Validation strategy (PLAN section 4): ask for the schema, retry once with the
validation error included, and if that also fails raise AgentError so the graph
can send the email to the human review queue instead of guessing.
"""

import time

from pydantic import ValidationError

from assistant.deps import Deps
from assistant.llm.base import LLMError, LLMResult, Message, T
from assistant.schemas import ToolCall, ToolResult
from assistant.tools import ALLOWLISTS, TOOLS


class AgentError(Exception):
    """An agent could not produce a valid answer; the graph routes to human review."""


def call_structured(deps: Deps, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
    try:
        return deps.llm.generate(system, user, schema, model)
    except (ValidationError, LLMError) as first_error:
        retry_user = f"{user}\n\nYour previous answer was invalid: {first_error}. Answer again with valid {schema.__name__}."
        try:
            return deps.llm.generate(system, retry_user, schema, model)
        except (ValidationError, LLMError) as second_error:
            raise AgentError(f"{schema.__name__} failed validation twice: {second_error}") from second_error


def execute_tool(deps: Deps, agent: str, call: ToolCall) -> ToolResult:
    """Run one tool call after checking the allowlist and validating its arguments.

    Problems are returned to the model as an error result rather than raised,
    so it can correct itself; each attempt still counts towards the call limit.
    """
    started = time.perf_counter()
    if call.name not in ALLOWLISTS[agent]:
        return ToolResult(name=call.name, error=f"tool '{call.name}' is not allowed for {agent}")
    tool = TOOLS[call.name]
    try:
        args = tool.spec.args_model.model_validate(call.args)
    except ValidationError as error:
        return ToolResult(name=call.name, error=f"invalid arguments: {error.errors(include_url=False)}")
    try:
        result = tool.run(deps, args)
    except Exception as error:  # a failing tool should not crash the whole run
        return ToolResult(name=call.name, error=f"{type(error).__name__}: {error}")
    return ToolResult(name=call.name, result=result, latency_ms=(time.perf_counter() - started) * 1000)


def run_tool_loop(
    deps: Deps, agent: str, system: str, user: str, schema: type[T], model: str
) -> tuple[LLMResult[T], list[ToolResult]]:
    """Let the model call allowed tools, at most settings.max_tool_calls times, then answer.

    Once the limit is reached the model is told it may not call tools, which
    forces a final answer. That bounds cost and makes a runaway loop impossible.
    """
    specs = [TOOLS[name].spec for name in ALLOWLISTS[agent]]
    max_calls = deps.settings.max_tool_calls
    messages = [Message(role="user", content=user)]
    results: list[ToolResult] = []
    retried = False

    while True:
        final = len(results) >= max_calls
        try:
            step = deps.llm.next_step(system, messages, specs, schema, model, final=final)
        except (ValidationError, LLMError) as error:
            if retried:
                raise AgentError(f"{agent}: {schema.__name__} failed validation twice: {error}") from error
            retried = True
            messages.append(Message(role="user", content=f"Your previous answer was invalid: {error}. Answer again."))
            continue

        if isinstance(step, LLMResult):
            return step, results
        if final:
            raise AgentError(f"{agent}: model asked for a tool after the limit of {max_calls} calls")

        result = execute_tool(deps, agent, step)
        results.append(result)
        messages.append(Message(role="tool_call", content=f"Call {step.name} {step.args}", tool_call=step))
        # The result carries the call too, so providers can match it to the call id.
        messages.append(Message(role="tool_result", content=result.result or f"Error: {result.error}", tool_call=step))
