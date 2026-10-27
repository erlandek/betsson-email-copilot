"""Learning loop: turn a human edit into durable preferences.

The input is only our own draft and the human's edited version, never the
incoming email, so this respects the memory write policy (DECISIONS.md, D30).
The output keys are a fixed list (PreferenceUpdate), so even a strange edit
cannot create arbitrary memory entries.
"""

from assistant.agents.structured import call_structured
from assistant.deps import Deps
from assistant.schemas import LearnedPreferences, PreferenceUpdate

SYSTEM = """The user edited a reply draft before sending it. Infer durable writing preferences from the
difference, only if they would apply to future replies too. Allowed keys: sign_off (the closing line, e.g.
"Best, Erland"), reply_length (short, medium or long), greeting (e.g. "Hi"), tone (e.g. friendly, formal).
Return an empty list if the edit was only about this specific email's content."""


def learn_from_edit(deps: Deps, before: str, after: str) -> list[PreferenceUpdate]:
    user = f"Draft before:\n<<<\n{before}\n>>>\nDraft after:\n<<<\n{after}\n>>>"
    result = call_structured(deps, SYSTEM, user, LearnedPreferences, deps.settings.model_for("triage"))
    return result.output.updates
