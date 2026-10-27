"""Command-line interface: process the inbox and approve, edit or reject each action.

    python scripts/copilot.py run            # interactive approvals
    python scripts/copilot.py run --approve-all
    python scripts/copilot.py memory         # what the copilot has learned
    python scripts/copilot.py outbox         # what was "sent" and scheduled (mock files)

State lives in runtime/ (checkpoints.db, memory.db, sent.json, calendar.json),
so a run can stop at an approval and continue later. Delete runtime/ to start over.
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import typer

from assistant.checkpoints import sqlite_checkpointer
from assistant.config import Settings
from assistant.deps import Deps, build_deps
from assistant.graph import build_graph, emails_to_process, pending_view, process_email
from assistant.schemas import HumanDecision

app = typer.Typer(add_completion=False, help="Single-user email copilot (mock mailbox and calendar).")


def setup() -> tuple[Deps, object]:
    settings = Settings()
    deps = build_deps(settings)
    return deps, build_graph(sqlite_checkpointer(settings.runtime_dir / "checkpoints.db"))


def render_view(view: dict) -> str:
    """The approval view as text: triage, summary, proposal and the facts the draft used."""
    t, p = view["triage"], view["proposal"]
    lines = [
        "=" * 72,
        f"Email {view['email']['id']} from {view['email']['from']}: {view['email']['subject']}",
        f"Triage: {t['intent']}, priority {t['priority']}, confidence {t['confidence']:.2f}, "
        f"tags {', '.join(t['labels']) or '-'}",
        f"Why: {t['rationale']}",
    ]
    if view.get("summary"):
        lines.append(f"Summary: {view['summary']['summary']}")
        lines += [f"  Open question: {q}" for q in view["summary"]["open_questions"]]
    payload = p["payload"]
    lines.append(f"Proposed action: {p['action_type']} ({p['reason']})")
    if p["action_type"] == "send_email":
        lines += [f"  To: {', '.join(payload['to'])}", f"  Subject: {payload['subject']}", "", payload["body"], ""]
    else:
        when = payload.get("start") or payload.get("due")
        lines.append(f"  {payload['kind']}: {payload['title']} at {when} {payload.get('attendees') or ''}")
    for fact in view["facts_used"]:
        lines.append(f"  Fact used [{fact['id']}]: {fact['text']}")
    if view.get("error"):
        lines.append(f"! {view['error']}")
    return "\n".join(lines)


def read_multiline(prompt: str) -> str:
    typer.echo(f"{prompt} (finish with a line containing only '.')")
    lines = []
    while (line := input()) != ".":
        lines.append(line)
    return "\n".join(lines)


def ask_human(view: dict, tz: ZoneInfo) -> HumanDecision | None:
    typer.echo(render_view(view))
    choice = typer.prompt("[a]pprove, [e]dit, [r]eject, [s]kip for now", default="s").strip().lower()[:1]
    if choice == "a":
        return HumanDecision(decision="approve")
    if choice == "r":
        return HumanDecision(decision="reject", comment=typer.prompt("Reason", default="") or None)
    if choice == "e":
        payload = dict(view["proposal"]["payload"])
        if view["proposal"]["action_type"] == "send_email":
            payload["body"] = read_multiline("New body")
        else:
            new = datetime.fromisoformat(typer.prompt("New local time (YYYY-MM-DD HH:MM)")).replace(tzinfo=tz)
            if payload["kind"] == "event":
                length = datetime.fromisoformat(payload["end"]) - datetime.fromisoformat(payload["start"])
                payload["start"], payload["end"] = new.isoformat(), (new + length).isoformat()
            else:
                payload["due"] = new.isoformat()
        comment = typer.prompt("Comment", default="") or None
        return HumanDecision(decision="edit", edited_payload=payload, comment=comment)
    return None  # skip: stays pending, run again later


@app.command()
def run(approve_all: bool = typer.Option(False, help="Approve every proposal without asking.")) -> None:
    """Process unread email; pause for approval before every outgoing action."""
    deps, graph = setup()
    tz = ZoneInfo(deps.settings.timezone)

    def decide(view: dict) -> HumanDecision | None:
        if approve_all:
            typer.echo(render_view(view))
            return HumanDecision(decision="approve")
        return ask_human(view, tz)

    try:
        for email in emails_to_process(deps):
            state = process_email(graph, deps, email.id, decide)
            if pending_view(state):
                status = "waiting for approval"
            elif state.get("route") == "human_review":
                status = f"needs review: {state['review_reason']}"
            else:
                status = state.get("outcome") or "labelled and archived"
            typer.echo(f"{email.id}  {email.subject[:45]:45}  {status}")
            for item in state.get("learned") or []:
                typer.echo(f"      learned preference {item.key} = {item.value!r}")
    finally:
        deps.obs.close()  # flush spans, write metrics.json
    typer.echo("\n" + deps.obs.metrics.summary())
    typer.echo(f"\nLogs, traces and metrics: {deps.settings.logs_dir}/ (scripts/show_trace.py prints a run)")


@app.command()
def memory() -> None:
    """Show preferences (seeded or learned) and recorded feedback."""
    deps, _ = setup()
    for p in deps.memory.preference_records():
        typer.echo(f"{p.key:24} {p.value!r:40} {p.source}")
    for f in deps.memory.all_feedback():
        typer.echo(f"feedback {f.email_id}: {f.decision} {('- ' + f.comment) if f.comment else ''}")


@app.command()
def outbox() -> None:
    """Show mock-sent emails and calendar entries created by the copilot."""
    settings = Settings()
    lines = []
    sent = settings.runtime_dir / "sent.json"
    for record in json.loads(sent.read_text(encoding="utf-8")) if sent.exists() else []:
        lines.append(f"SENT {record['action_id']}  to {', '.join(record['to'])}: {record['subject']}")
    calendar = settings.runtime_dir / "calendar.json"
    if calendar.exists():
        data = json.loads(calendar.read_text(encoding="utf-8"))
        for item in [*data["events"], *data["tasks"]]:
            if item.get("action_id"):
                lines.append(f"CALENDAR {item['action_id']}  {item['title']} {item.get('start') or item.get('due')}")
    typer.echo("\n".join(lines) or "Nothing sent or scheduled yet. Approve an action with `copilot.py run` first.")


if __name__ == "__main__":
    app()
