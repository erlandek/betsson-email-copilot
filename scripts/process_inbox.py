"""Run every email that needs attention through the graph and print a triage table.

    python scripts/process_inbox.py

Uses LLM_PROVIDER from .env (fake by default). Nothing is approved or sent:
every run stops at the approval step. Use scripts/copilot.py run to decide.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant.config import Settings  # noqa: E402
from assistant.deps import build_deps  # noqa: E402
from assistant.graph import build_graph, emails_to_process, process_email  # noqa: E402


def describe(state) -> str:
    if state.get("route") == "human_review":
        return f"Human review: {state['review_reason']}"
    if "proposal" in state:
        p = state["proposal"]
        if p.action_type.value == "send_email":
            return f"Awaiting approval: reply to {', '.join(p.payload.to)} (facts {', '.join(p.payload.facts_used) or 'none'})"
        when = p.payload.start or p.payload.due
        return f"Awaiting approval: {p.action_type.value} '{p.payload.title}' {when:%a %d %b %H:%M}"
    return "Labelled and archived"


def main() -> None:
    settings = Settings()
    # A throwaway runtime folder keeps demo runs independent of each other.
    settings.runtime_dir = Path(tempfile.mkdtemp(prefix="copilot-"))
    deps = build_deps(settings)
    app = build_graph()

    rows = []
    for email in emails_to_process(deps):
        state = process_email(app, deps, email.id)
        t = state.get("triage")
        rows.append((
            email.id, email.sender[:30], email.subject[:38],
            t.intent.value if t else "-", t.priority.value if t else "-", f"{t.confidence:.2f}" if t else "-",
            ", ".join(deps.mail.get_email(email.id).labels), describe(state),
        ))

    headers = ("id", "from", "subject", "intent", "priority", "conf", "labels", "outcome")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, tuple("-" * w for w in widths), *rows]:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
