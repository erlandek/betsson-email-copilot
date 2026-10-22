"""Summariser: turn a thread into a ThreadSummary. One LLM call, no tools needed."""

from assistant.agents.prompts import UNTRUSTED_NOTE, format_thread
from assistant.agents.structured import call_structured
from assistant.deps import Deps
from assistant.schemas import Thread, ThreadSummary

SYSTEM = f"""Summarise this email thread for the user, who will reply to it.
summary: two sentences at most. key_points: one line per message, oldest first.
open_questions: questions in the latest message that still need an answer. action_items: what the user must do.
{UNTRUSTED_NOTE}"""


def summarise_thread(deps: Deps, thread: Thread) -> ThreadSummary:
    return call_structured(deps, SYSTEM, format_thread(thread), ThreadSummary, deps.settings.model_for("draft")).output
