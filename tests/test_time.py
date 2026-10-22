"""Time zones and working hours."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from conftest import DATA_DIR
from pydantic import ValidationError

from assistant.agents.scheduler import check_proposal, default_due, next_week_window, propose_schedule
from assistant.agents.structured import AgentError
from assistant.schemas import Email, Intent, ScheduleProposal

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValidationError):
        Email(id="x", thread_id="t", sender="a@b.c", recipients=["me@b.c"], subject="s", body="b",
              received_at=datetime(2026, 10, 21, 9, 0))
    with pytest.raises(ValidationError):
        ScheduleProposal(kind="task", title="x", due=datetime(2026, 10, 30, 9, 0), reason="r")


def test_sample_data_uses_stockholm_offsets():
    # Before 25 October Stockholm is UTC+2 (CEST), after it UTC+1 (CET).
    for e in json.loads((DATA_DIR / "inbox.json").read_text(encoding="utf-8")):
        at = datetime.fromisoformat(e["received_at"])
        assert at.utcoffset() == at.astimezone(STOCKHOLM).utcoffset(), e["id"]
    for ev in json.loads((DATA_DIR / "calendar.json").read_text(encoding="utf-8"))["events"]:
        at = datetime.fromisoformat(ev["start"])
        assert at.utcoffset() == at.astimezone(STOCKHOLM).utcoffset(), ev["id"]


def test_next_week_window_crosses_the_dst_change_correctly():
    start, end = next_week_window(datetime(2026, 10, 21, 9, 15, tzinfo=timezone(timedelta(hours=2))), STOCKHOLM)
    assert start.isoformat() == "2026-10-26T00:00:00+01:00"  # Monday after the switch to CET
    assert end.isoformat() == "2026-10-31T00:00:00+01:00"


def test_reminder_is_due_at_0900_local_after_the_dst_change(deps):
    # e09 arrived on 20 October (CEST); a week later Stockholm is on CET, and 09:00 must stay 09:00.
    received = datetime(2026, 10, 20, 16, 40, tzinfo=timezone(timedelta(hours=2)))
    assert default_due(received, STOCKHOLM).isoformat() == "2026-10-27T09:00:00+01:00"
    proposal, _ = propose_schedule(deps, deps.mail.get_email("e09"), Intent.follow_up)
    assert proposal.kind == "task"
    assert proposal.due.astimezone(STOCKHOLM).strftime("%H:%M") == "09:00"


def with_hours(deps, hours: str):
    deps.memory.set_preference("working_hours", hours, source="seeded")
    return deps


def test_free_time_respects_working_hours_from_preferences(deps):
    from assistant.agents.structured import execute_tool
    from assistant.schemas import ToolCall

    with_hours(deps, "10:30-15:00")
    result = execute_tool(deps, "scheduler", ToolCall(name="check_availability", args={
        "start": "2026-10-26T00:00:00+01:00", "end": "2026-10-31T00:00:00+01:00"}))
    periods = [line.removeprefix("Free time: ").split(" | ") for line in result.result.splitlines()]
    assert periods
    for start, end in periods:
        s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
        assert s.astimezone(STOCKHOLM).time() >= datetime(1, 1, 1, 10, 30).time()
        assert e.astimezone(STOCKHOLM).time() <= datetime(1, 1, 1, 15, 0).time()


def test_scheduled_meeting_is_inside_working_hours(deps):
    with_hours(deps, "10:00-16:00")
    proposal, _ = propose_schedule(deps, deps.mail.get_email("e07"), Intent.meeting_request)
    start, end = proposal.start.astimezone(STOCKHOLM), proposal.end.astimezone(STOCKHOLM)
    assert start.time().hour >= 10 and (end.hour, end.minute) <= (16, 0)


def test_check_proposal_rejects_out_of_hours_weekend_and_clashes(deps):
    def event(day, hour, minute=0):
        start = datetime(2026, 10, day, hour, minute, tzinfo=STOCKHOLM)
        return ScheduleProposal(kind="event", title="x", start=start, end=start + timedelta(minutes=30), reason="r")

    check_proposal(deps, event(27, 15))  # Tuesday 15:00 is free
    for bad in (event(27, 18), event(31, 11), event(27, 13, 30)):  # evening, Saturday, product review
        with pytest.raises(AgentError):
            check_proposal(deps, bad)
