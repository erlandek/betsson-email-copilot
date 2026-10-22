import json
from datetime import datetime, timezone

import pytest
from conftest import DATA_DIR
from pydantic import ValidationError

from assistant.schemas import (
    ActionProposal,
    DraftReply,
    Email,
    HumanDecision,
    Intent,
    ScheduleProposal,
    TriageResult,
)

NOW = datetime(2026, 10, 27, 15, 0, tzinfo=timezone.utc)
DRAFT = DraftReply(to=["a@example.com"], subject="Re: hi", body="Hello")
EVENT = ScheduleProposal(kind="event", title="Sync", start=NOW, end=NOW.replace(hour=15, minute=30), reason="r")
TASK = ScheduleProposal(kind="task", title="Follow up", due=NOW, reason="r")


def test_sample_inbox_validates_and_has_headers():
    raw = json.loads((DATA_DIR / "inbox.json").read_text(encoding="utf-8"))
    emails = [Email.model_validate(e) for e in raw]
    assert len(emails) >= 15
    assert any("List-Unsubscribe" in e.headers for e in emails)


def test_email_headers_default_to_empty():
    email = Email(id="x", thread_id="t", sender="a@b.c", recipients=["me@b.c"], subject="s", body="b",
                  received_at=NOW)
    assert email.headers == {}
    assert email.archived is False


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        TriageResult(intent="fyi", priority="low", confidence=0.9, rationale="r", mood="happy")


def test_triage_rejects_unknown_intent_and_bad_confidence():
    with pytest.raises(ValidationError):
        TriageResult(intent="party_invite", priority="low", confidence=0.9, rationale="r")
    with pytest.raises(ValidationError):
        TriageResult(intent="fyi", priority="low", confidence=1.5, rationale="r")
    assert TriageResult(intent="fyi", priority="low", confidence=0.7, rationale="r").intent is Intent.fyi


def test_event_needs_valid_times_and_task_needs_due():
    with pytest.raises(ValidationError):
        ScheduleProposal(kind="event", title="x", start=NOW, reason="r")
    with pytest.raises(ValidationError):
        ScheduleProposal(kind="event", title="x", start=NOW, end=NOW, reason="r")
    with pytest.raises(ValidationError):
        ScheduleProposal(kind="task", title="x", reason="r")


def test_draft_needs_recipient_and_body():
    with pytest.raises(ValidationError):
        DraftReply(to=[], subject="s", body="b")
    with pytest.raises(ValidationError):
        DraftReply(to=["a@b.c"], subject="s", body="")


def test_action_payload_must_match_type():
    assert ActionProposal(action_type="send_email", payload=DRAFT, reason="r").requires_approval
    ActionProposal(action_type="create_event", payload=EVENT, reason="r")
    ActionProposal(action_type="create_task", payload=TASK, reason="r")
    with pytest.raises(ValidationError):
        ActionProposal(action_type="send_email", payload=EVENT, reason="r")
    with pytest.raises(ValidationError):
        ActionProposal(action_type="create_event", payload=TASK, reason="r")


def test_action_proposal_round_trips_through_json():
    # The checkpointer stores state as JSON, so the union payload must come back as the right type.
    proposal = ActionProposal(action_type="create_event", payload=EVENT, reason="r")
    restored = ActionProposal.model_validate_json(proposal.model_dump_json())
    assert isinstance(restored.payload, ScheduleProposal)
    assert restored == proposal


def test_edit_decision_needs_payload():
    with pytest.raises(ValidationError):
        HumanDecision(decision="edit")
    assert HumanDecision(decision="edit", edited_payload=DRAFT).edited_payload == DRAFT
    assert HumanDecision(decision="reject", comment="wrong tone").edited_payload is None
