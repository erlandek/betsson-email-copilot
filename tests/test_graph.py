import pytest

from assistant.graph import NEEDS_REVIEW_LABEL, build_graph, emails_to_process, process_email
from assistant.schemas import ActionType


@pytest.fixture
def run(deps):
    app = build_graph()
    return lambda email_id: process_email(app, deps, email_id)


def test_emails_to_process_is_latest_unread_inbound_per_thread(deps):
    ids = [e.id for e in emails_to_process(deps)]
    assert len(ids) == 10  # 15 emails, 10 threads, one email each
    assert "e01" not in ids and "e03" in ids  # older message in the same thread is skipped
    assert all(deps.mail.get_email(i).sender != "erland@example.com" for i in ids)


def test_read_latest_message_or_own_reply_is_skipped(deps):
    deps.mail.get_email("e15").unread = False  # user already read it
    assert "e15" not in [e.id for e in emails_to_process(deps)]
    deps.mail.get_email("e06").received_at = deps.mail.get_email("e07").received_at.replace(hour=23)  # own reply last
    assert "e07" not in [e.id for e in emails_to_process(deps)]


def test_reply_route_proposes_a_send_that_needs_approval(run):
    state = run("e04")
    assert state["route"] == "reply"
    assert state["proposal"].action_type == ActionType.send_email
    assert state["proposal"].requires_approval
    assert state["summary"].summary


def test_meeting_route_proposes_event_in_a_free_slot(run, deps):
    state = run("e07")
    proposal = state["proposal"]
    assert proposal.action_type == ActionType.create_event
    event = proposal.payload
    assert event.start.strftime("%A") in ("Tuesday", "Wednesday") and event.start.hour >= 13
    assert deps.calendar.is_free(event.start, event.end)
    assert deps.calendar.find_by_action_id("anything") is None  # proposed only: nothing executed yet


def test_follow_up_route_proposes_task(run):
    assert run("e09")["proposal"].action_type == ActionType.create_task


@pytest.mark.parametrize("email_id", ["e10", "e11", "e12"])
def test_newsletter_fyi_and_spam_are_labelled_and_archived(run, deps, email_id):
    state = run(email_id)
    assert state["route"] == "archive" and "proposal" not in state
    email = deps.mail.get_email(email_id)
    assert email.archived and email.labels  # archived, never deleted


def test_spam_and_newsletters_get_a_category_label(run, deps):
    run("e12")
    run("e10")
    assert "Spam" in deps.mail.get_email("e12").labels
    assert "Newsletters" in deps.mail.get_email("e10").labels


@pytest.mark.parametrize("email_id", ["e13", "e14"])
def test_fraud_and_ambiguous_go_to_human_review(run, deps, email_id):
    state = run(email_id)
    assert state["route"] == "human_review" and "proposal" not in state
    assert NEEDS_REVIEW_LABEL in deps.mail.get_email(email_id).labels


def test_fraud_review_reason_explains_why(run):
    assert "fraud" in run("e13")["review_reason"].lower()


def test_failed_worker_goes_to_human_review(run, deps):
    from assistant.llm.base import LLMError

    class BrokenDrafts(type(deps.llm)):
        def next_step(self, *args, **kwargs):
            raise LLMError("provider down")

    deps.llm = BrokenDrafts()
    state = run("e04")
    assert state["route"] == "human_review" and state["errors"]


def test_graph_is_deterministic(deps):
    app = build_graph()
    first = process_email(app, deps, "e04")["proposal"]
    second = process_email(app, deps, "e04")["proposal"]
    assert first == second
