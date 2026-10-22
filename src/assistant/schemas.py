"""Pydantic schemas for every message between agents, tools and the human.

Enums and Literals keep LLM outputs bounded: a model cannot invent a new intent
or decision, because validation rejects anything outside the allowed values.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


# Every datetime must carry a time zone (AwareDatetime rejects naive values), so
# working hours and scheduling are never off by an hour around DST changes.


class Schema(BaseModel):
    # Reject unknown fields so a malformed LLM answer fails loudly instead of
    # silently carrying extra data through the graph.
    model_config = ConfigDict(extra="forbid")


# --- Mail -------------------------------------------------------------------


class Email(Schema):
    id: str
    thread_id: str
    sender: str
    sender_name: str | None = None
    recipients: list[str]
    cc: list[str] = []
    subject: str
    body: str
    received_at: AwareDatetime
    labels: list[str] = []
    # Raw headers such as List-Unsubscribe. The newsletter pre-filter reads them,
    # so obvious newsletters never reach the LLM.
    headers: dict[str, str] = {}
    unread: bool = True
    archived: bool = False


class Thread(Schema):
    thread_id: str
    subject: str
    messages: list[Email]  # oldest first

    @property
    def latest(self) -> Email:
        return self.messages[-1]


# --- Triage -----------------------------------------------------------------


class Intent(str, Enum):
    reply_needed = "reply_needed"
    meeting_request = "meeting_request"
    follow_up = "follow_up"
    fyi = "fyi"
    newsletter = "newsletter"
    spam = "spam"


class Priority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class TriageResult(Schema):
    intent: Intent
    priority: Priority
    labels: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# --- Worker outputs ---------------------------------------------------------


class ThreadSummary(Schema):
    summary: str
    key_points: list[str] = []
    open_questions: list[str] = []
    action_items: list[str] = []


class DraftReply(Schema):
    to: list[str] = Field(min_length=1)
    subject: str
    body: str = Field(min_length=1)
    tone: str = "neutral"
    # IDs of org facts the draft relied on, e.g. ["F02"]. Shown to the human at
    # approval time so they can check the draft is grounded.
    facts_used: list[str] = []


class ScheduleProposal(Schema):
    kind: Literal["event", "task"]
    title: str
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None
    due: AwareDatetime | None = None
    attendees: list[str] = []
    reason: str

    @model_validator(mode="after")
    def check_times(self) -> "ScheduleProposal":
        if self.kind == "event":
            if self.start is None or self.end is None:
                raise ValueError("an event needs start and end")
            if self.end <= self.start:
                raise ValueError("event end must be after start")
        elif self.due is None:
            raise ValueError("a task needs a due date")
        return self


# --- Actions and human decisions -------------------------------------------


class ActionType(str, Enum):
    # Only outgoing actions. Labelling and archiving are internal and automatic,
    # and there is deliberately no delete action.
    send_email = "send_email"
    create_event = "create_event"
    create_task = "create_task"


class ActionProposal(Schema):
    action_type: ActionType
    payload: DraftReply | ScheduleProposal
    reason: str
    requires_approval: bool = True

    @model_validator(mode="after")
    def payload_matches_type(self) -> "ActionProposal":
        if self.action_type == ActionType.send_email:
            ok = isinstance(self.payload, DraftReply)
        else:
            expected = "event" if self.action_type == ActionType.create_event else "task"
            ok = isinstance(self.payload, ScheduleProposal) and self.payload.kind == expected
        if not ok:
            raise ValueError(f"payload does not match action_type {self.action_type.value}")
        return self


class HumanDecision(Schema):
    decision: Literal["approve", "edit", "reject"]
    # An edited payload is validated against the same schemas as the proposal.
    edited_payload: DraftReply | ScheduleProposal | None = None
    comment: str | None = None

    @model_validator(mode="after")
    def edit_needs_payload(self) -> "HumanDecision":
        if self.decision == "edit" and self.edited_payload is None:
            raise ValueError("an edit decision needs edited_payload")
        return self


# --- Tools ------------------------------------------------------------------


class ToolCall(Schema):
    name: str
    args: dict[str, Any] = {}


class ToolResult(Schema):
    name: str
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


# --- Records written by the mock adapters ----------------------------------
# Each carries the action_id created by the execute node, so a resumed run can
# check whether the action already happened before doing it again.


class SentRecord(Schema):
    action_id: str
    thread_id: str
    to: list[str]
    subject: str
    body: str
    sent_at: AwareDatetime


class CalendarEvent(Schema):
    id: str
    title: str
    start: AwareDatetime
    end: AwareDatetime
    attendees: list[str] = []
    action_id: str | None = None  # None for events that were already in the calendar


class CalendarTask(Schema):
    id: str
    title: str
    due: AwareDatetime
    action_id: str | None = None


# --- Long-term memory records ----------------------------------------------


class PreferenceRecord(Schema):
    key: str  # e.g. "sign_off", "reply_length"
    value: str
    source: Literal["seeded", "learned"]
    updated_at: AwareDatetime


class ContactRecord(Schema):
    email: str
    name: str | None = None
    relationship: str | None = None
    vip: bool = False
    notes: str | None = None


class FeedbackRecord(Schema):
    email_id: str
    decision: Literal["approve", "edit", "reject"]
    draft_before: str
    draft_after: str | None = None
    comment: str | None = None
    created_at: AwareDatetime
