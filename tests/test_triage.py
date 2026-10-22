from datetime import datetime, timezone

from assistant.agents.triage import FRAUD_LABEL, SPAM_OVERRIDE_LABEL, apply_priority_rules, is_newsletter, triage_email
from assistant.schemas import ContactRecord, Email, Intent, Priority, TriageResult

VIP = ContactRecord(email="boss@example.com", name="Boss", vip=True)
COLLEAGUE = ContactRecord(email="pal@example.com", name="Pal")


def make_email(body="Hello", sender="someone@else.example", subject="Hi", headers=None) -> Email:
    return Email(id="x", thread_id="t", sender=sender, recipients=["erland@example.com"], subject=subject, body=body,
                 received_at=datetime(2026, 10, 21, tzinfo=timezone.utc), headers=headers or {})


def result(intent="reply_needed", priority="medium", confidence=0.9) -> TriageResult:
    return TriageResult(intent=intent, priority=priority, labels=[intent], confidence=confidence, rationale="r")


# --- is_newsletter -------------------------------------------------------------


def test_newsletter_by_header_sender_or_memory(deps):
    assert is_newsletter(make_email(headers={"List-Unsubscribe": "<x>"}), deps.memory)
    assert is_newsletter(make_email(sender="no-reply@shop.example"), deps.memory)
    assert is_newsletter(make_email(sender="newsletter@aiweekly.example"), deps.memory)
    assert is_newsletter(make_email(sender="news@cloudtools.example"), deps.memory)  # known in seed memory
    assert not is_newsletter(make_email(sender="priya.nair@example.com"), deps.memory)


def test_newsletter_skips_the_llm(deps, mail):
    triage, llm_call = triage_email(deps, deps.mail.get_email("e10"))
    assert triage.intent == Intent.newsletter and triage.confidence == 1.0
    assert llm_call is None


# --- apply_priority_rules ------------------------------------------------------


def test_vip_bumps_one_level():
    assert apply_priority_rules(result(priority="medium"), make_email(sender=VIP.email), VIP).priority == Priority.high


def test_deadline_bumps_one_level():
    email = make_email(body="I need this by Friday please.")
    assert apply_priority_rules(result(priority="low"), email, COLLEAGUE).priority == Priority.medium


def test_vip_and_deadline_bump_two_levels():
    email = make_email(body="Urgent: numbers needed today.", sender=VIP.email)
    assert apply_priority_rules(result(priority="low"), email, VIP).priority == Priority.high


def test_newsletter_and_spam_are_always_low():
    email = make_email(body="URGENT deal ends by Friday!")
    assert apply_priority_rules(result("spam", "high"), email, None).priority == Priority.low
    assert apply_priority_rules(result("newsletter", "high"), email, None).priority == Priority.low


def test_known_contact_is_never_spam():
    out = apply_priority_rules(result("spam", "low", 0.9), make_email(sender=COLLEAGUE.email), COLLEAGUE)
    assert out.intent != Intent.spam
    assert out.confidence <= 0.5 and SPAM_OVERRIDE_LABEL in out.labels


def test_unknown_sender_with_bank_change_is_capped_and_flagged():
    email = make_email(body="Our bank details have changed, please use the new IBAN.")
    out = apply_priority_rules(result(confidence=0.95), email, None)
    assert out.confidence == 0.5
    assert FRAUD_LABEL in out.labels


def test_known_contact_with_bank_language_is_not_flagged():
    email = make_email(body="Can you check the IBAN on the invoice?", sender=COLLEAGUE.email)
    out = apply_priority_rules(result(confidence=0.95), email, COLLEAGUE)
    assert out.confidence == 0.95 and FRAUD_LABEL not in out.labels


def test_rules_do_not_mutate_input():
    original = result(priority="low")
    apply_priority_rules(original, make_email(body="urgent"), VIP)
    assert original.priority == Priority.low


# --- On the sample inbox ---------------------------------------------------------


def test_vip_with_deadline_is_high(deps):
    triage, _ = triage_email(deps, deps.mail.get_email("e03"))
    assert triage.intent == Intent.reply_needed and triage.priority == Priority.high


def test_lookalike_bank_email_goes_to_review(deps):
    triage, _ = triage_email(deps, deps.mail.get_email("e13"))
    assert triage.confidence <= 0.5
    assert FRAUD_LABEL in triage.labels
