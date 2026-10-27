"""What the human sees before approving, and how a decision becomes the final action.

The approval view shows the triage result, the thread summary,
the proposal, and the facts the draft relied on, so the human can judge both
the action and the reasoning behind it.
"""

import hashlib

from assistant.deps import Deps
from assistant.schemas import ActionProposal, DraftReply, HumanDecision, ScheduleProposal


def build_view(state: dict, deps: Deps) -> dict:
    """A JSON-friendly snapshot for the interrupt. Everything a reviewer needs, nothing more."""
    proposal: ActionProposal = state["proposal"]
    facts = []
    if isinstance(proposal.payload, DraftReply):
        for fact_id in proposal.payload.facts_used:
            fact = deps.knowledge.get(fact_id)
            facts.append({"id": fact_id, "text": fact.text if fact else "(unknown fact id)"})
    email, summary = state["email"], state.get("summary")
    return {
        "email": {"id": email.id, "from": email.sender, "subject": email.subject},
        "triage": state["triage"].model_dump(mode="json"),
        "summary": summary.model_dump(mode="json") if summary else None,
        "proposal": proposal.model_dump(mode="json"),
        "facts_used": facts,
    }


def final_payload(state: dict) -> DraftReply | ScheduleProposal:
    decision: HumanDecision | None = state.get("decision")
    if decision and decision.decision == "edit":
        return decision.edited_payload
    return state["proposal"].payload


def check_decision(proposal: ActionProposal, decision: HumanDecision) -> None:
    """An edit must still be a valid action of the same type (raises ValidationError if not)."""
    if decision.decision == "edit":
        ActionProposal(action_type=proposal.action_type, payload=decision.edited_payload, reason=proposal.reason)


def make_action_id(email_id: str, proposal: ActionProposal, payload: DraftReply | ScheduleProposal) -> str:
    """Deterministic id for an approved action (DECISIONS.md, D27).

    The same approved action always gets the same id, so a crash and retry, or a
    resumed run, finds the earlier record instead of acting twice. An edited
    payload is a different action and gets a different id.
    """
    digest = hashlib.sha256(f"{email_id}|{proposal.action_type.value}|{payload.model_dump_json()}".encode())
    return f"act-{digest.hexdigest()[:16]}"


def describe(payload: DraftReply | ScheduleProposal) -> str:
    """Short text form of a payload, stored as feedback. Drafts are our own text, never the incoming email."""
    if isinstance(payload, DraftReply):
        return payload.body
    when = payload.start or payload.due
    return f"{payload.kind}: {payload.title} at {when.isoformat()}"
