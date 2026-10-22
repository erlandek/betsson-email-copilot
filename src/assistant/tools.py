"""Tools the worker agents may call, each with a Pydantic model for its arguments.

The model never calls our code directly: it proposes a ToolCall, and
agents/tool_loop.py checks the allowlist, validates the arguments with the
tool's args model, runs it and turns the result into text for the transcript.
All tools here are read-only. Outgoing actions (send, create event) are never
tools: they only happen in the execute node after human approval.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assistant.deps import Deps
from assistant.llm.base import ToolSpec


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupContactArgs(Args):
    email: str = Field(min_length=3)


class SearchKnowledgeArgs(Args):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=3, ge=1, le=5)


class GetPreferencesArgs(Args):
    pass


class GetThreadArgs(Args):
    thread_id: str


class CheckAvailabilityArgs(Args):
    start: datetime
    end: datetime
    duration_minutes: int = Field(default=30, ge=15, le=240)

    @model_validator(mode="after")
    def check_window(self) -> "CheckAvailabilityArgs":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.end - self.start > timedelta(days=14):
            raise ValueError("search window is limited to 14 days")
        return self


@dataclass
class Tool:
    spec: ToolSpec
    run: Callable[[Deps, BaseModel], str]  # returns text for the transcript


# --- Implementations ---------------------------------------------------------
# Results are short "Key: value" lines: easy for a model to read, and the same
# format the FakeLLM parses.


def _lookup_contact(deps: Deps, args: LookupContactArgs) -> str:
    contact = deps.memory.get_contact(args.email)
    if not contact:
        return f"Contact {args.email}: unknown sender"
    vip = ", VIP" if contact.vip else ""
    return f"Contact {args.email}: {contact.name} ({contact.relationship}{vip})"


def _search_knowledge(deps: Deps, args: SearchKnowledgeArgs) -> str:
    facts = deps.knowledge.search(args.query, args.k)
    return "\n".join(f"[{f.id}] {f.text}" for f in facts) or "No matching facts."


def _get_preferences(deps: Deps, args: GetPreferencesArgs) -> str:
    return "\n".join(f"Preference {key}: {value}" for key, value in deps.memory.get_preferences().items())


def _get_thread(deps: Deps, args: GetThreadArgs) -> str:
    from assistant.agents.prompts import format_thread  # local import avoids a cycle

    return format_thread(deps.mail.get_thread(args.thread_id))


def _check_availability(deps: Deps, args: CheckAvailabilityArgs) -> str:
    hours = deps.memory.get_preferences().get("working_hours", "09:00-17:00")
    day_start, day_end = (int(part.split(":")[0]) for part in hours.split("-"))
    slots = deps.calendar.free_slots(args.start, args.end, timedelta(minutes=args.duration_minutes), day_start, day_end)
    # Merge back-to-back slots into free periods: a week fits in a few lines
    # instead of dozens, which saves tokens and is easier for a model to read.
    periods: list[list[datetime]] = []
    for start, end in slots:
        if periods and start <= periods[-1][1]:
            periods[-1][1] = end
        else:
            periods.append([start, end])
    lines = [f"Free time: {s.isoformat()} | {e.isoformat()}" for s, e in periods]
    return "\n".join(lines) or "No free time in this window."


TOOLS: dict[str, Tool] = {
    tool.spec.name: tool
    for tool in [
        Tool(ToolSpec(name="lookup_contact", description="Look up a sender in the contact book.",
                      args_model=LookupContactArgs), _lookup_contact),
        Tool(ToolSpec(name="search_knowledge", description="Search organisational facts; cite them by ID.",
                      args_model=SearchKnowledgeArgs), _search_knowledge),
        Tool(ToolSpec(name="get_preferences", description="Get the user's writing and scheduling preferences.",
                      args_model=GetPreferencesArgs), _get_preferences),
        Tool(ToolSpec(name="get_thread", description="Get every message in an email thread.",
                      args_model=GetThreadArgs), _get_thread),
        Tool(ToolSpec(name="check_availability", description="List free calendar slots in a time window.",
                      args_model=CheckAvailabilityArgs), _check_availability),
    ]
}

# Least privilege: each agent sees only the tools it needs.
ALLOWLISTS: dict[str, list[str]] = {
    "drafter": ["get_thread", "search_knowledge", "get_preferences", "lookup_contact"],
    "scheduler": ["check_availability", "get_preferences"],
}
