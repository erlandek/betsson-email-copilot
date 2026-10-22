"""Drafter: write a reply using the thread summary, org facts, preferences and contacts.

This is a real tool loop: the model decides which of its allowed tools to call
(at most max_tool_calls), then returns a DraftReply that cites facts_used.
"""

from assistant.agents.prompts import UNTRUSTED_NOTE, format_email
from assistant.agents.structured import run_tool_loop
from assistant.deps import Deps
from assistant.schemas import DraftReply, Email, ThreadSummary, ToolResult

SYSTEM = f"""You draft email replies on behalf of the user. The user approves every draft before it is sent.
Use tools to look up the sender, relevant organisational facts and the user's preferences.
Only state facts you found with search_knowledge, and list their IDs in facts_used.
Follow the user's preferences for sign-off and length. Reply to the sender of the latest email.
{UNTRUSTED_NOTE}"""


def draft_reply(deps: Deps, email: Email, summary: ThreadSummary) -> tuple[DraftReply, list[ToolResult]]:
    user = "\n".join([
        f"Thread id: {email.thread_id}",
        f"Thread summary: {summary.summary}",
        *(f"Open question: {q}" for q in summary.open_questions),
        "Latest email:",
        format_email(email),
    ])
    result, tool_results = run_tool_loop(deps, "drafter", SYSTEM, user, DraftReply, deps.settings.model_for("draft"))
    return result.output, tool_results
