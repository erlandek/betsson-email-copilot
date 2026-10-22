from datetime import datetime, timedelta, timezone

import pytest

from assistant.adapters import MockMailAdapter
from assistant.schemas import CalendarEvent, CalendarTask, DraftReply, ScheduleProposal

CET = timezone(timedelta(hours=1))
DRAFT = DraftReply(to=["priya.nair@example.com"], subject="Re: Sync", body="Tuesday 15:00 works.")


# --- Mail -------------------------------------------------------------------


def test_list_emails_sorted_oldest_first(mail):
    emails = mail.list_emails()
    assert len(emails) == 15
    assert [e.received_at for e in emails] == sorted(e.received_at for e in emails)


def test_get_thread_returns_messages_in_order(mail):
    thread = mail.get_thread("t-q4-kpis")
    assert [m.id for m in thread.messages] == ["e01", "e02", "e03"]
    assert thread.latest.id == "e03"


def test_unknown_ids_raise(mail):
    with pytest.raises(KeyError):
        mail.get_email("nope")
    with pytest.raises(KeyError):
        mail.get_thread("nope")


def test_newsletter_has_list_unsubscribe_header(mail):
    assert "List-Unsubscribe" in mail.get_email("e10").headers


def test_apply_label_is_idempotent(mail):
    mail.apply_label("e12", "spam")
    mail.apply_label("e12", "spam")
    assert mail.get_email("e12").labels == ["spam"]


def test_archive_hides_but_never_deletes(mail):
    mail.archive("e12")
    assert "e12" not in [e.id for e in mail.list_emails()]
    assert mail.get_email("e12").archived  # still there
    assert not hasattr(mail, "delete")


def test_label_changes_do_not_touch_the_sample_file(mail, tmp_path):
    from conftest import DATA_DIR

    mail.apply_label("e12", "spam")
    fresh = MockMailAdapter(DATA_DIR / "inbox.json", tmp_path / "other.json")
    assert fresh.get_email("e12").labels == []


def test_send_writes_record_with_action_id(mail, tmp_path):
    record = mail.send(DRAFT, "t-roadmap-sync", action_id="act-1")
    assert record.action_id == "act-1"
    assert (tmp_path / "sent.json").exists()
    assert mail.find_sent("act-1") == record
    assert mail.find_sent("act-2") is None


def test_send_same_action_id_twice_sends_once(mail, tmp_path):
    first = mail.send(DRAFT, "t-roadmap-sync", action_id="act-1")
    second = mail.send(DRAFT, "t-roadmap-sync", action_id="act-1")
    assert first == second
    # A new adapter instance (as after a restart) still sees exactly one record.
    from conftest import DATA_DIR

    restarted = MockMailAdapter(DATA_DIR / "inbox.json", tmp_path / "sent.json")
    assert restarted.find_sent("act-1") == first
    assert len(restarted._load_sent()) == 1


# --- Calendar ---------------------------------------------------------------


def test_is_free_detects_conflicts(calendar):
    review_start = datetime(2026, 10, 27, 13, 30, tzinfo=CET)
    assert not calendar.is_free(review_start, review_start + timedelta(minutes=30))
    assert calendar.is_free(datetime(2026, 10, 27, 15, 0, tzinfo=CET), datetime(2026, 10, 27, 15, 30, tzinfo=CET))


def test_free_slots_skip_busy_times_and_respect_working_hours(calendar):
    day = datetime(2026, 10, 27, 0, 0, tzinfo=CET)
    slots = calendar.free_slots(day, day + timedelta(days=1), timedelta(minutes=30))
    starts = [s.strftime("%H:%M") for s, _ in slots]
    assert starts[:2] == ["09:00", "10:00"]  # 09:30 clashes with the stand-up (09:30 to 09:45)
    assert "13:00" not in starts and "14:30" not in starts  # product review 13:00 to 15:00
    assert "15:00" in starts
    assert starts[-1] == "16:30"


def test_free_slots_skip_weekends(calendar):
    saturday = datetime(2026, 10, 31, 0, 0, tzinfo=CET)
    assert calendar.free_slots(saturday, saturday + timedelta(days=2), timedelta(minutes=30)) == []


def test_create_event_is_idempotent_and_persisted(calendar):
    start = datetime(2026, 10, 27, 15, 0, tzinfo=CET)
    proposal = ScheduleProposal(kind="event", title="Roadmap sync", start=start, end=start + timedelta(minutes=30),
                                attendees=["priya.nair@example.com"], reason="r")
    first = calendar.create_event(proposal, action_id="act-9")
    second = calendar.create_event(proposal, action_id="act-9")
    assert first == second
    assert isinstance(calendar.find_by_action_id("act-9"), CalendarEvent)
    assert not calendar.is_free(start, start + timedelta(minutes=30))
    assert len([e for e in calendar.list_events(start, start + timedelta(hours=1)) if e.action_id]) == 1


def test_create_task_is_idempotent(calendar):
    proposal = ScheduleProposal(kind="task", title="Follow up on DPA", due=datetime(2026, 10, 31, 9, 0, tzinfo=CET),
                                reason="r")
    first = calendar.create_task(proposal, action_id="act-10")
    assert calendar.create_task(proposal, action_id="act-10") == first
    assert isinstance(calendar.find_by_action_id("act-10"), CalendarTask)


def test_create_event_rejects_task_proposal(calendar):
    task = ScheduleProposal(kind="task", title="x", due=datetime(2026, 10, 31, tzinfo=CET), reason="r")
    with pytest.raises(ValueError):
        calendar.create_event(task, action_id="act-11")
