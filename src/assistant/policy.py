"""Approval policy: which outgoing actions need a human decision.

APPROVAL_MODE=all (the default) asks for approval on everything, as the brief
requires. risk_based shows how this would relax in production: actions that are
private and reversible could run automatically, while anything that leaves the
company or touches other people's calendars still needs a human.
"""

from assistant.schemas import ActionProposal, ActionType, Priority, ScheduleProposal, TriageResult


def requires_approval(action: ActionProposal, triage: TriageResult, mode: str = "all") -> bool:
    if mode == "all":
        return True
    # risk_based: the documented production option, not the default.
    if action.action_type == ActionType.send_email:
        return True  # external and irreversible
    if action.action_type == ActionType.create_event:
        return isinstance(action.payload, ScheduleProposal) and bool(action.payload.attendees)  # visible to others
    # A private reminder is low risk, unless the email itself was important.
    return triage.priority == Priority.high
