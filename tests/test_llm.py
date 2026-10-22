import pytest

from assistant.config import Settings
from assistant.llm import FakeLLM, LLMError, get_llm_client
from assistant.schemas import DraftReply, Intent, ScheduleProposal, ThreadSummary, TriageResult

# What the fake classifier should say about each inbound sample email, before
# any rule-based adjustments (those arrive with the triage agent in Phase 2).
EXPECTED_INTENT = {
    "e03": Intent.reply_needed,
    "e04": Intent.reply_needed,
    "e07": Intent.meeting_request,
    "e09": Intent.follow_up,
    "e10": Intent.newsletter,
    "e11": Intent.fyi,
    "e12": Intent.spam,
    "e15": Intent.reply_needed,
}


def as_prompt(email) -> str:
    return f"From: {email.sender}\nSubject: {email.subject}\nBody: {email.body}"


def triage(email) -> TriageResult:
    return FakeLLM().generate("Classify.", as_prompt(email), TriageResult, "fake-triage").output


def test_sample_data_covers_every_intent():
    assert set(EXPECTED_INTENT.values()) == set(Intent)


@pytest.mark.parametrize("email_id,intent", EXPECTED_INTENT.items())
def test_fake_triage_on_sample_inbox(mail, email_id, intent):
    result = triage(mail.get_email(email_id))
    assert result.intent == intent
    assert result.confidence >= 0.6


def test_ambiguous_email_has_low_confidence(mail):
    assert triage(mail.get_email("e14")).confidence < 0.6


def test_fake_is_deterministic(mail):
    email = mail.get_email("e13")
    assert triage(email) == triage(email)


def test_result_carries_usage_numbers(mail):
    result = FakeLLM().generate("Classify.", as_prompt(mail.get_email("e04")), TriageResult, "fake-triage")
    assert result.provider == "fake" and result.model == "fake-triage"
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_fake_summary(mail):
    thread = mail.get_thread("t-q4-kpis")
    prompt = "\n".join(as_prompt(m) for m in thread.messages)
    summary = FakeLLM().generate("Summarise.", prompt, ThreadSummary, "m").output
    assert "3 message(s)" in summary.summary
    assert summary.open_questions


def test_fake_draft_uses_facts_and_preferences(mail):
    prompt = as_prompt(mail.get_email("e04")) + "\n[F04] Partner support hours are weekdays 09:00 to 17:00 CET."
    draft = FakeLLM().generate("Draft.", prompt, DraftReply, "m").output
    assert draft.to == ["mark.jensen@nordwind.example"]
    assert draft.subject.startswith("Re: ")
    assert draft.facts_used == ["F04"]

    styled = FakeLLM().generate("Draft.", prompt + "\nPreference sign_off: Best, Erland\nPreference reply_length: short",
                                DraftReply, "m").output
    assert styled.body.endswith("Best, Erland")
    assert styled.tone == "concise"


def test_fake_schedule_event_and_task():
    event = FakeLLM().generate(
        "Schedule.", "From: priya.nair@example.com\nSubject: Re: Sync\nFree time: 2026-10-27T15:00:00+01:00 | 2026-10-27T17:00:00+01:00",
        ScheduleProposal, "m").output
    assert event.kind == "event" and event.title == "Sync"
    assert (event.end - event.start).seconds == 30 * 60  # default meeting length, not the whole free period
    task = FakeLLM().generate("Schedule.", "Subject: DPA\nDue: 2026-10-31T09:00:00+01:00", ScheduleProposal, "m").output
    assert task.kind == "task"
    with pytest.raises(LLMError):
        FakeLLM().generate("Schedule.", "Subject: nothing to go on", ScheduleProposal, "m")


def test_fake_rejects_unknown_schema():
    with pytest.raises(LLMError):
        FakeLLM().generate("s", "u", Settings, "m")


def test_factory_defaults_to_fake_and_real_providers_need_keys():
    settings = Settings(_env_file=None)
    assert isinstance(get_llm_client(settings), FakeLLM)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm_client(settings, provider="openai")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm_client(settings, provider="anthropic")

