"""Scheduler: propose a calendar event (meeting request) or a reminder task (follow-up).

Like the drafter, it runs a bounded tool loop. The search window is computed in
code (next working week in the user's time zone) so the model never has to do
date arithmetic, which models are unreliable at.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from assistant.agents.prompts import UNTRUSTED_NOTE, format_email
from assistant.agents.structured import AgentError, run_tool_loop
from assistant.deps import Deps
from assistant.schemas import Email, Intent, ScheduleProposal, ToolResult
from assistant.tools import working_hours

SYSTEM = f"""You schedule things for the user. The user approves every proposal before it is created.
For a meeting request, call check_availability in the given search window and propose an event
(kind "event") in a free slot that fits what the sender asked for, with the sender as attendee.
For a follow-up, propose a reminder task (kind "task") with a sensible due date; no attendees.
Give a one-sentence reason.
{UNTRUSTED_NOTE}"""


def next_week_window(received_at: datetime, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Monday 00:00 to Saturday 00:00 of the week after the email, in local time."""
    local = received_at.astimezone(tz)
    monday = (local + timedelta(days=7 - local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return monday, monday + timedelta(days=5)


def propose_schedule(deps: Deps, email: Email, intent: Intent) -> tuple[ScheduleProposal, list[ToolResult]]:
    lines = [f"Intent: {intent.value}"]
    if intent == Intent.meeting_request:
        start, end = next_week_window(email.received_at, ZoneInfo(deps.settings.timezone))
        lines.append(f"Search window: {start.isoformat()} | {end.isoformat()}")
    else:
        lines.append("Propose a reminder task, not an event.")
    lines += ["Latest email:", format_email(email)]
    result, tool_results = run_tool_loop(
        deps, "scheduler", SYSTEM, "\n".join(lines), ScheduleProposal, deps.settings.model_for("draft")
    )
    check_proposal(deps, result.output)
    return result.output, tool_results


def check_proposal(deps: Deps, proposal: ScheduleProposal) -> None:
    """Verify an event in code instead of trusting the model: weekday, working hours, no clash."""
    if proposal.kind != "event":
        return
    tz = ZoneInfo(deps.settings.timezone)
    start, end = proposal.start.astimezone(tz), proposal.end.astimezone(tz)
    day_start, day_end = working_hours(deps)
    if start.weekday() >= 5 or start.date() != end.date() or start.time() < day_start or end.time() > day_end:
        raise AgentError(f"proposed event {start:%a %H:%M}-{end:%H:%M} is outside working hours")
    if not deps.calendar.is_free(proposal.start, proposal.end):
        raise AgentError(f"proposed event {start:%a %H:%M} clashes with the calendar")
