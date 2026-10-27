"""The orchestrator: a LangGraph StateGraph that runs one email through the workers.

Routing is deterministic code on the triage result (a conditional edge), not an
LLM supervisor: the most important decision in the flow stays predictable,
testable and free. The workers are where the agentic behaviour lives.

    fetch -> triage -> router -> human_review                 (low confidence or failure)
                              -> archive                      (newsletter, spam, fyi)
                              -> summarise -> draft -> propose_action   (reply_needed)
                              -> schedule -> propose_action             (meeting_request, follow_up)

    propose_action -> human_approval (interrupt: approve / edit / reject)
                   -> execute (approve or edit) -> update_memory
                   -> update_memory (reject)

The graph pauses at human_approval. A checkpointer saves the state, so the
decision can arrive minutes or days later, even in a new process.
"""

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypedDict

import structlog
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from assistant.agents.drafter import draft_reply
from assistant.agents.learner import learn_from_edit
from assistant.agents.scheduler import propose_schedule
from assistant.agents.structured import AgentError
from assistant.agents.summariser import summarise_thread
from assistant.agents.triage import FRAUD_LABEL, LOOKALIKE_LABEL, triage_email
from assistant.approval import build_view, check_decision, describe, final_payload, make_action_id
from assistant.checkpoints import serializer
from assistant.deps import Deps
from assistant.policy import requires_approval
from assistant.schemas import (
    ActionProposal,
    ActionType,
    ContactRecord,
    DraftReply,
    Email,
    FeedbackRecord,
    HumanDecision,
    Intent,
    PreferenceUpdate,
    ScheduleProposal,
    ThreadSummary,
    ToolResult,
    TriageResult,
)

NEEDS_REVIEW_LABEL = "Needs review"
ARCHIVE_INTENTS = {Intent.newsletter, Intent.spam, Intent.fyi}
# Mailbox categories for what gets archived, so spam and newsletters stay findable.
CATEGORY_LABELS = {Intent.spam: "Spam", Intent.newsletter: "Newsletters"}


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
    decision: HumanDecision
    action_id: str
    outcome: str  # what actually happened, in words
    learned: list[PreferenceUpdate]
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
        return {"errors": [str(error)], "review_reason": "Triage failed after a retry; see errors.jsonl."}
    # Topic tags and rule warnings are low risk and reversible, so they are
    # applied automatically, even when the intent itself goes to human review.
    for label in result.labels:
        deps.mail.apply_label(email.id, label)
    deps.obs.metrics.inc("emails_by_intent_total", intent=result.intent.value)
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
    elif not reason and LOOKALIKE_LABEL in result.labels:
        reason = "Sender's domain imitates a known contact's domain. Verify the sender."
    elif not reason:
        reason = f"Low confidence ({result.confidence:.2f}) for {result.intent.value}."
    return {"route": "human_review", "review_reason": reason}


def archive(state: State, runtime: Runtime[Deps]) -> State:
    # Label and archive only; there is no delete (DECISIONS.md, D5).
    mail, email = runtime.context.mail, state["email"]
    category = CATEGORY_LABELS.get(state["triage"].intent)
    if category:
        mail.apply_label(email.id, category)
    mail.archive(email.id)
    return {"route": "archive"}


def summarise(state: State, runtime: Runtime[Deps]) -> State:
    deps = runtime.context
    try:
        return {"summary": summarise_thread(deps, deps.mail.get_thread(state["email"].thread_id))}
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Summary failed after a retry; see errors.jsonl."}


def draft(state: State, runtime: Runtime[Deps]) -> State:
    try:
        reply, tool_results = draft_reply(runtime.context, state["email"], state["summary"])
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Drafting failed after a retry; see errors.jsonl."}
    return {"draft": reply, "tool_results": tool_results}


def schedule(state: State, runtime: Runtime[Deps]) -> State:
    try:
        proposal, tool_results = propose_schedule(runtime.context, state["email"], state["triage"].intent)
    except AgentError as error:
        return {"errors": [str(error)], "review_reason": "Scheduling failed after a retry; see errors.jsonl."}
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


def human_approval(state: State, runtime: Runtime[Deps]) -> State:
    """Pause until a human approves, edits or rejects the proposal.

    interrupt() saves the state and stops the run. When the run is resumed with
    Command(resume=decision), LangGraph runs this node again from the top and
    interrupt() returns the decision instead of pausing. So nothing before
    interrupt() may have side effects.
    """
    view = build_view(state, runtime.context)
    while True:
        raw = interrupt(view)
        try:
            decision = HumanDecision.model_validate(raw)
            check_decision(state["proposal"], decision)
            return {"decision": decision}
        except ValidationError as error:
            # Ask again rather than act on an invalid edit.
            view = {**view, "error": f"Invalid decision: {error.errors(include_url=False)}"}


def route_after_proposal(state: State) -> str:
    return "human_approval" if state["proposal"].requires_approval else "execute"


def route_after_approval(state: State) -> str:
    return "update_memory" if state["decision"].decision == "reject" else "execute"


def execute(state: State, runtime: Runtime[Deps]) -> State:
    """Carry out the approved action exactly once (DECISIONS.md, D27).

    The action_id is derived from the email and the final payload before acting,
    and the adapters are asked for it first. A resumed or retried run finds the
    earlier record and does nothing.
    """
    deps, proposal, email = runtime.context, state["proposal"], state["email"]
    payload = final_payload(state)
    action_id = make_action_id(email.id, proposal, payload)

    if proposal.action_type == ActionType.send_email:
        done = deps.mail.find_sent(action_id)
        if not done:
            deps.mail.send(payload, email.thread_id, action_id)
        what = f"email to {', '.join(payload.to)}"
    else:
        done = deps.calendar.find_by_action_id(action_id)
        if not done:
            create = deps.calendar.create_event if payload.kind == "event" else deps.calendar.create_task
            create(payload, action_id)
        what = f"{payload.kind} '{payload.title}'"
    outcome = f"Already done earlier: {what}" if done else f"Done: {what}"
    return {"action_id": action_id, "outcome": outcome}


def update_memory(state: State, runtime: Runtime[Deps]) -> State:
    """Write to long-term memory, only from the human decision (DECISIONS.md, D30)."""
    deps, proposal = runtime.context, state["proposal"]
    decision = state.get("decision")
    if decision is None:  # executed without approval (risk_based mode): nothing was decided
        return {}

    before = describe(proposal.payload)
    after = describe(decision.edited_payload) if decision.decision == "edit" else None
    deps.memory.add_feedback(FeedbackRecord(
        email_id=state["email"].id, decision=decision.decision, draft_before=before, draft_after=after,
        comment=decision.comment, created_at=datetime.now(timezone.utc),
    ))

    update: State = {}
    if decision.decision == "reject":
        update["outcome"] = f"Rejected: {decision.comment or 'no reason given'}"
    is_reply = proposal.action_type == ActionType.send_email
    if is_reply and decision.decision != "reject":
        # Replying to someone is the user's decision to correspond with them.
        for address in final_payload(state).to:
            if deps.memory.get_contact(address) is None:
                deps.memory.upsert_contact(ContactRecord(email=address, relationship="replied to"))
    if is_reply and decision.decision == "edit":
        try:
            learned = learn_from_edit(deps, before, after)
        except AgentError as error:
            return {**update, "errors": [*state.get("errors", []), str(error)]}
        for item in learned:
            deps.memory.set_preference(item.key, item.value, source="learned")
        update["learned"] = learned
    return update


# --- Graph -------------------------------------------------------------------


def traced(name: str, node: Callable) -> Callable:
    """Run a node inside its own span, with a log line and a latency metric.

    Errors a node handles itself (an AgentError that routes to human review)
    come back in the state's "errors" list; they are logged and counted here,
    and the span is marked as failed, without stopping the run.
    """

    def run(state: State, runtime: Runtime[Deps]) -> State:
        obs = runtime.context.obs
        with obs.tracer.start_as_current_span(f"node {name}", record_exception=False,
                                              set_status_on_exception=False) as span:
            started = time.perf_counter()
            try:
                update = node(state, runtime)
            except GraphBubbleUp:  # interrupt(): waiting for a human, not a failure
                span.set_attribute("assistant.waiting_for_human", True)
                obs.log.info("node_paused", node=name)
                raise
            except Exception as error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                obs.metrics.inc("errors_total", node=name)
                obs.log.exception("node_failed", node=name)
                raise
            finally:
                latency_ms = (time.perf_counter() - started) * 1000
                obs.metrics.observe("node_latency_ms", latency_ms, node=name)

            new_errors = [e for e in update.get("errors", []) if e not in state.get("errors", [])]
            for message in new_errors:
                obs.metrics.inc("errors_total", node=name)
                obs.log.error("node_error", node=name, error=message)
            if new_errors:
                span.set_status(Status(StatusCode.ERROR, new_errors[0][:200]))
            obs.log.info("node", node=name, latency_ms=round(latency_ms, 1), route=update.get("route"))
            return update

    return run


def build_graph(checkpointer=None):
    """Compile the graph. Without a checkpointer an in-memory one is used (tests, one-off runs)."""
    graph = StateGraph(State, context_schema=Deps)
    for name, node in [
        ("fetch", fetch), ("triage", triage), ("human_review", human_review), ("archive", archive),
        ("summarise", summarise), ("draft", draft), ("schedule", schedule), ("propose_action", propose_action),
        ("human_approval", human_approval), ("execute", execute), ("update_memory", update_memory),
    ]:
        graph.add_node(name, traced(name, node))

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "triage")
    graph.add_conditional_edges("triage", route_after_triage, ["human_review", "archive", "summarise", "schedule"])
    graph.add_conditional_edges("summarise", lambda s: "human_review" if s.get("review_reason") else "draft",
                                ["human_review", "draft"])
    graph.add_conditional_edges("draft", route_after_worker, ["human_review", "propose_action"])
    graph.add_conditional_edges("schedule", route_after_worker, ["human_review", "propose_action"])
    graph.add_conditional_edges("propose_action", route_after_proposal, ["human_approval", "execute"])
    graph.add_conditional_edges("human_approval", route_after_approval, ["execute", "update_memory"])
    graph.add_edge("execute", "update_memory")
    for terminal in ("human_review", "archive", "update_memory"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=serializer()))


def emails_to_process(deps: Deps) -> list[Email]:
    """One email per thread: the latest message, if it is unread and someone else sent it.

    Earlier messages are context, not separate work (the triage and summary see
    the whole thread). A thread whose latest message is the user's own reply is
    waiting on the other side, so it is skipped.
    """
    latest: dict[str, Email] = {}
    for email in deps.mail.list_emails():  # oldest first, so later messages overwrite earlier ones
        latest[email.thread_id] = email
    todo = [e for e in latest.values() if e.unread and e.sender != deps.settings.user_email]
    return sorted(todo, key=lambda e: e.received_at)


Decide = Callable[[dict], HumanDecision | None]


def process_email(app, deps: Deps, email_id: str, decide: Decide | None = None) -> State:
    """Run one email to the end, or until it waits for a human.

    Each email is its own checkpointer thread (thread_id = email id), so:
    - a finished email is not processed again;
    - an email waiting for approval is resumed, even from a new process.
    `decide` gets the approval view and returns a HumanDecision, or None to
    leave it pending. Without `decide`, runs stop at the approval step.
    """
    # recursion_limit is LangGraph's step cap: a hard stop if routing ever looped.
    config = {"configurable": {"thread_id": email_id}, "recursion_limit": 25}
    snapshot = app.get_state(config)
    if snapshot.values and not snapshot.next:
        return snapshot.values  # finished earlier

    obs, run_id = deps.obs, uuid.uuid4().hex[:12]
    # One trace per run. A run resumed later, in another process, starts a new trace
    # with the same email_id, so the two can still be found together.
    with (
        obs.tracer.start_as_current_span("email_run", attributes={"assistant.email_id": email_id,
                                                                  "assistant.run_id": run_id}) as span,
        structlog.contextvars.bound_contextvars(run_id=run_id, email_id=email_id),
    ):
        if snapshot.next:
            result = {**snapshot.values, "__interrupt__": snapshot.interrupts}  # waiting for a decision
            span.set_attribute("assistant.resumed", True)
        else:
            result = app.invoke({"email_id": email_id}, config, context=deps)

        while result.get("__interrupt__") and decide:
            decision = decide(result["__interrupt__"][0].value)
            if decision is None:
                break
            obs.metrics.inc("decisions_total", decision=decision.decision)
            obs.log.info("human_decision", decision=decision.decision)
            result = app.invoke(Command(resume=decision.model_dump(mode="json")), config, context=deps)

        status = "waiting_for_human" if result.get("__interrupt__") else "done"
        span.set_attributes({"assistant.route": result.get("route", "-"), "assistant.status": status})
        obs.metrics.inc("email_runs_total", route=result.get("route", "-"), status=status)
        obs.log.info("email_run", route=result.get("route"), status=status, outcome=result.get("outcome"))
    return result


def pending_view(state: State) -> dict | None:
    """The approval view if this run is waiting for a human, else None."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value if interrupts else None
