"""Human approval, idempotent execution, short- and long-term memory, learning loop."""

import json

import pytest

from assistant.checkpoints import sqlite_checkpointer
from assistant.deps import build_deps
from assistant.graph import build_graph, pending_view, process_email
from assistant.schemas import HumanDecision, PreferenceUpdate, SentRecord

SHORT_BODY = "Hi Mark,\n\nSandbox credentials come via the partner portal.\n\nBest, Erland"


@pytest.fixture
def app():
    return build_graph()


def approve(view):
    return HumanDecision(decision="approve")


def edit_body(body):
    def decide(view):
        payload = {**view["proposal"]["payload"], "body": body}
        return HumanDecision(decision="edit", edited_payload=payload, comment="shorter")
    return decide


def sent(deps) -> list[dict]:
    path = deps.settings.runtime_dir / "sent.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def test_run_pauses_before_sending_with_a_full_approval_view(app, deps):
    state = process_email(app, deps, "e04")
    view = pending_view(state)
    assert view is not None and sent(deps) == []
    assert view["triage"]["intent"] == "reply_needed"
    assert view["summary"]["summary"]
    assert view["proposal"]["action_type"] == "send_email"
    assert view["facts_used"] and all(f["text"] != "(unknown fact id)" for f in view["facts_used"])


def test_approve_sends_once_with_an_action_id(app, deps):
    state = process_email(app, deps, "e04", approve)
    assert state["outcome"].startswith("Done")
    records = sent(deps)
    assert len(records) == 1 and records[0]["action_id"] == state["action_id"]


def test_reject_sends_nothing_and_records_feedback(app, deps):
    state = process_email(app, deps, "e04", lambda v: HumanDecision(decision="reject", comment="not now"))
    assert sent(deps) == [] and state["outcome"] == "Rejected: not now"
    assert deps.memory.all_feedback()[0].decision == "reject"


def test_skip_leaves_it_pending_and_a_new_process_can_resume(deps, tmp_path):
    db = tmp_path / "checkpoints.db"
    first = process_email(build_graph(sqlite_checkpointer(db)), deps, "e07", lambda v: None)
    assert pending_view(first)

    # A new process: new graph, new adapters, same runtime files.
    deps2 = build_deps(deps.settings)
    resumed = process_email(build_graph(sqlite_checkpointer(db)), deps2, "e07", approve)
    assert resumed["outcome"].startswith("Done: event")
    assert deps2.calendar.find_by_action_id(resumed["action_id"])


def test_finished_email_is_not_processed_again(app, deps):
    process_email(app, deps, "e04", approve)
    again = process_email(app, deps, "e04", approve)
    assert again["outcome"].startswith("Done") and len(sent(deps)) == 1


def test_execute_checks_the_action_id_before_acting(app, deps):
    """Simulate a crash after sending but before the checkpoint was saved."""
    from assistant.approval import make_action_id

    view_state = process_email(app, deps, "e04")
    proposal = view_state["proposal"]
    action_id = make_action_id("e04", proposal, proposal.payload)
    deps.mail.send(proposal.payload, "t-partner-api", action_id)  # the "earlier attempt"

    state = process_email(app, deps, "e04", approve)
    assert state["outcome"].startswith("Already done earlier")
    assert len(sent(deps)) == 1


def test_invalid_edit_is_asked_again(app, deps):
    answers = iter([
        HumanDecision(decision="edit", edited_payload={"kind": "task", "title": "x", "due": "2026-10-30T09:00:00+01:00",
                                                       "reason": "r"}),
        HumanDecision(decision="approve"),
    ])
    seen = []

    def decide(view):
        seen.append(view)
        return next(answers)

    state = process_email(app, deps, "e04", decide)
    assert "error" in seen[1] and state["outcome"].startswith("Done")


def test_edit_sends_the_edited_text_and_learns_preferences(app, deps):
    state = process_email(app, deps, "e04", edit_body(SHORT_BODY))
    assert sent(deps)[0]["body"] == SHORT_BODY
    assert {(p.key, p.value) for p in state["learned"]} == {("sign_off", "Best, Erland"), ("reply_length", "short")}
    prefs = {p.key: p for p in deps.memory.preference_records()}
    assert prefs["sign_off"].source == "learned"

    # The learning loop: a later, similar email is drafted in the new style.
    later = pending_view(process_email(app, deps, "e15"))
    body = later["proposal"]["payload"]["body"]
    assert body.endswith("Best, Erland") and len(later["facts_used"]) == 1


def test_approved_reply_adds_the_recipient_as_a_contact(app, deps):
    assert deps.memory.get_contact("anna.holm@fjordanalytics.example") is None
    process_email(app, deps, "e15", approve)
    assert deps.memory.get_contact("anna.holm@fjordanalytics.example").relationship == "replied to"


def test_email_content_alone_never_writes_memory(app, deps):
    before_prefs, before_domains = deps.memory.get_preferences(), deps.memory.known_domains()
    for email_id in ["e13", "e14", "e15", "e10", "e12"]:
        process_email(app, deps, email_id)  # no decisions
    assert deps.memory.get_preferences() == before_prefs
    assert deps.memory.known_domains() == before_domains
    assert deps.memory.get_contact("accounts@nordwind-billing.example") is None
    assert deps.memory.all_feedback() == []


def test_learned_keys_are_restricted():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PreferenceUpdate(key="trusted_iban", value="GB33 BUKB")


def test_risk_based_mode_runs_a_private_task_without_approval(deps):
    deps.settings.approval_mode = "risk_based"
    state = process_email(build_graph(), deps, "e09")
    assert pending_view(state) is None
    assert state["outcome"].startswith("Done: task")


def test_risk_based_mode_still_asks_before_sending(deps):
    deps.settings.approval_mode = "risk_based"
    assert pending_view(process_email(build_graph(), deps, "e04"))


def test_sent_records_validate(app, deps):
    process_email(app, deps, "e04", approve)
    SentRecord.model_validate(sent(deps)[0])
