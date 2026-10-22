"""The orchestrator: a LangGraph StateGraph that runs one email through the workers.

Routing is deterministic code on the triage result (a conditional edge), not an
LLM supervisor: the most important decision in the flow stays predictable,
testable and free. The workers are where the agentic behaviour lives.

    fetch -> triage -> router -> human_review                 (low confidence or failure)
                              -> archive                      (newsletter, spam, fyi)
                              -> summarise -> draft -> propose_action   (reply_needed)
                              -> schedule -> propose_action             (meeting_request, follow_up)

Phase 3 adds human approval (interrupt), execution and memory updates after propose_action.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from assistant.agents.drafter import draft_reply
from assistant.agents.scheduler import propose_schedule
from assistant.agents.structured import AgentError
from assistant.agents.summariser import summarise_thread
from assistant.agents.triage import FRAUD_LABEL, triage_email
from assistant.deps import Deps
from assistant.policy import requires_approval
from assistant.schemas import (
    ActionProposal,
    ActionType,
    DraftReply,
    Email,
    Intent,
    ScheduleProposal,
    ThreadSummary,
    ToolResult,
    TriageResult,
)

NEEDS_REVIEW_LABEL = "Needs review"
ARCHIVE_INTENTS = {Intent.newsletter, Intent.spam, Intent.fyi}


class State(TypedDict, total=False):
    email_id: str  # the only input
    email: Email
    triage: TriageResult
    summary: ThreadSummary
    draft: DraftReply
    schedule: ScheduleProposal
    tool_results: list[ToolResult]
    proposal: ActionProposal
    route: str  # which branch the router chose, for logs and the demo table
    review_reason: str
    errors: list[str]


# --- Nodes -------------------------------------------------------------------
# Each node takes the state and returns only the keys it changes.


def fetch(state: State, runtime: Runtime[Deps]) -> State:
    return {"email": runtime.context.mail.get_email(state["email_id"])}


def triage(state: State, runtime: Runtime[Deps]) -> State:
    deps, email = runtime.context, state["email"]
    try:
        result, _ = triage_email(deps, email)
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Triage failed validation."}
    # Labelling is low risk and reversible, so it happens automatically. An
    # unsure intent label would mislead, so below the threshold only rule labels
    # (such as the fraud warning) are applied, and human_review adds its own.
    confident = result.confidence >= deps.settings.confidence_threshold
    for label in result.labels:
        if confident or label != result.intent.value:
            deps.mail.apply_label(email.id, label)
    return {"triage": result}


def route_after_triage(state: State, runtime: Runtime[Deps]) -> str:
    result = state.get("triage")
    if result is None:
        return "human_review"
    if result.confidence < runtime.context.settings.confidence_threshold:
        return "human_review"
    if result.intent in ARCHIVE_INTENTS:
        return "archive"
    if result.intent == Intent.reply_needed:
        return "summarise"
    return "schedule"  # meeting_request or follow_up


def human_review(state: State, runtime: Runtime[Deps]) -> State:
    runtime.context.mail.apply_label(state["email"].id, NEEDS_REVIEW_LABEL)
    reason = state.get("review_reason")
    result = state.get("triage")
    if not reason and FRAUD_LABEL in result.labels:
        reason = "Possible fraud: unknown sender asks to change payment details. Verify by phone (F05)."
    elif not reason:
        reason = f"Low confidence ({result.confidence:.2f}) for {result.intent.value}."
    return {"route": "human_review", "review_reason": reason}


def archive(state: State, runtime: Runtime[Deps]) -> State:
    # Archive only; there is no delete (amendment A4).
    runtime.context.mail.archive(state["email"].id)
    return {"route": "archive"}


def summarise(state: State, runtime: Runtime[Deps]) -> State:
    deps = runtime.context
    try:
        return {"summary": summarise_thread(deps, deps.mail.get_thread(state["email"].thread_id))}
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Summary failed validation."}


def draft(state: State, runtime: Runtime[Deps]) -> State:
    try:
        reply, tool_results = draft_reply(runtime.context, state["email"], state["summary"])
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Draft failed validation."}
    return {"draft": reply, "tool_results": tool_results}


def schedule(state: State, runtime: Runtime[Deps]) -> State:
    try:
        proposal, tool_results = propose_schedule(runtime.context, state["email"], state["triage"].intent)
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Scheduling failed validation."}
    return {"schedule": proposal, "tool_results": tool_results}


def route_after_worker(state: State) -> str:
    return "human_review" if state.get("review_reason") else "propose_action"


def propose_action(state: State, runtime: Runtime[Deps]) -> State:
    if "draft" in state:
        action_type, payload, route = ActionType.send_email, state["draft"], "reply"
        reason = "Reply drafted for an email that needs an answer."
    else:
        payload = state["schedule"]
        action_type = ActionType.create_event if payload.kind == "event" else ActionType.create_task
        route, reason = "schedule", payload.reason
    proposal = ActionProposal(action_type=action_type, payload=payload, reason=reason)
    needs_approval = requires_approval(proposal, state["triage"], runtime.context.settings.approval_mode)
    return {"proposal": proposal.model_copy(update={"requires_approval": needs_approval}), "route": route}


# --- Graph -------------------------------------------------------------------


def build_graph():
    graph = StateGraph(State, context_schema=Deps)
    for name, node in [
        ("fetch", fetch), ("triage", triage), ("human_review", human_review), ("archive", archive),
        ("summarise", summarise), ("draft", draft), ("schedule", schedule), ("propose_action", propose_action),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "triage")
    graph.add_conditional_edges("triage", route_after_triage, ["human_review", "archive", "summarise", "schedule"])
    graph.add_conditional_edges("summarise", lambda s: "human_review" if s.get("review_reason") else "draft",
                                ["human_review", "draft"])
    graph.add_conditional_edges("draft", route_after_worker, ["human_review", "propose_action"])
    graph.add_conditional_edges("schedule", route_after_worker, ["human_review", "propose_action"])
    for terminal in ("human_review", "archive", "propose_action"):
        graph.add_edge(terminal, END)
    return graph.compile()


def emails_to_process(deps: Deps) -> list[Email]:
    """The latest email of each thread, if someone else sent it: that is the one that may need action."""
    latest: dict[str, Email] = {}
    for email in deps.mail.list_emails():  # oldest first, so later messages overwrite earlier ones
        latest[email.thread_id] = email
    inbound = [e for e in latest.values() if e.sender != deps.settings.user_email]
    return sorted(inbound, key=lambda e: e.received_at)


def process_email(app, deps: Deps, email_id: str) -> State:
    # recursion_limit is LangGraph's step cap: a hard stop if routing ever looped.
    return app.invoke({"email_id": email_id}, context=deps, config={"recursion_limit": 25})
