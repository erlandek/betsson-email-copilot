"""A deterministic stand-in for a real LLM.

FakeLLM lets the whole demo and test suite run with no API keys and identical
output on every run. It does not try to be clever: it picks a small handler per
output schema and applies keyword rules to the prompt text.

Prompt convention the handlers rely on (agents in later phases build prompts
this way anyway, to keep untrusted email text clearly separated):
    From: <address>          Subject: <subject>          Body: ...
    [F02] <org fact text>    Preference sign_off: <text>
    Free slot: <iso start> | <iso end>    Due: <iso datetime>
"""

import re
from datetime import datetime

from assistant.llm.base import LLMClient, LLMError, LLMResult, T
from assistant.schemas import DraftReply, Intent, Priority, ScheduleProposal, ThreadSummary, TriageResult

# Keyword cues per intent. Order matters on ties: earlier intents win, and a tie
# also lowers confidence so the email is sent to human review.
INTENT_KEYWORDS: dict[Intent, list[str]] = {
    Intent.spam: ["congratulations", "you have been selected", "claim your", "prize", "act now"],
    Intent.newsletter: ["unsubscribe", "newsletter", "this week in", "weekly digest", "issue #"],
    Intent.meeting_request: ["set up", "30 minutes", "meeting", "call", "invite", "calendar", "afternoon works"],
    Intent.follow_up: ["get back to you", "come back to you", "by end of next week", "reminder", "circle back"],
    Intent.fyi: ["fyi", "no action needed", "for your information", "maintenance", "heads up"],
    Intent.reply_needed: ["?", "could you", "can you", "please send", "please update", "let me know"],
}

LOW_PRIORITY_INTENTS = {Intent.newsletter, Intent.spam, Intent.fyi}


def _field(text: str, name: str) -> str | None:
    match = re.search(rf"^{name}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _bodies(text: str) -> list[str]:
    # A body runs from "Body:" until the next message's "From:" line or the end.
    return [b.strip() for b in re.findall(r"^Body:\s*(.*?)(?=^From:|\Z)", text, flags=re.MULTILINE | re.DOTALL)]


def _first_sentence(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[0].rstrip().endswith(","):  # skip a greeting such as "Hi Erland,"
        lines = lines[1:]
    sentence = re.split(r"(?<=[.!?])\s", " ".join(lines), maxsplit=1)[0]
    return sentence[:200]


class FakeLLM(LLMClient):
    provider = "fake"

    def generate(self, system: str, user: str, schema: type[T], model: str) -> LLMResult[T]:
        handlers = {
            TriageResult: self._triage,
            ThreadSummary: self._summary,
            DraftReply: self._draft,
            ScheduleProposal: self._schedule,
        }
        if schema not in handlers:
            raise LLMError(f"FakeLLM has no handler for {schema.__name__}")
        output = handlers[schema](user)
        # Word counts stand in for tokens so cost and token metrics have numbers.
        return LLMResult[schema](
            output=output,
            provider=self.provider,
            model=model,
            input_tokens=len((system + " " + user).split()),
            output_tokens=len(output.model_dump_json().split()),
        )

    def _triage(self, text: str) -> TriageResult:
        lowered = text.lower()
        scores = {intent: sum(lowered.count(k) for k in keys) for intent, keys in INTENT_KEYWORDS.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)  # stable: keeps dict order on ties
        (top, top_score), (_, second_score) = ranked[0], ranked[1]

        if top_score == 0:
            intent, confidence = Intent.fyi, 0.4
        else:
            intent = top
            # More separation between the top two intents means more confidence.
            confidence = min(0.95, 0.5 + 0.15 * (top_score - second_score))
        priority = Priority.low if intent in LOW_PRIORITY_INTENTS else Priority.medium
        matched = [k for k in INTENT_KEYWORDS.get(intent, []) if k in lowered]
        return TriageResult(
            intent=intent,
            priority=priority,
            labels=[intent.value],
            confidence=round(confidence, 2),
            rationale=f"Keyword match for {intent.value}: {matched or 'none'}",
        )

    def _summary(self, text: str) -> ThreadSummary:
        subject = _field(text, "Subject") or "(no subject)"
        senders = sorted(set(re.findall(r"^From:\s*(.+)$", text, flags=re.MULTILINE)))
        bodies = _bodies(text)
        # Questions from the latest message are the ones still open.
        questions = [q.strip() for q in re.findall(r"[^.!?\n]*\?", bodies[-1])] if bodies else []
        return ThreadSummary(
            summary=f"Thread '{subject}' with {len(bodies)} message(s) from {', '.join(senders) or 'unknown'}.",
            key_points=[_first_sentence(b) for b in bodies],
            open_questions=questions,
            action_items=["Reply to the latest message"] if questions else [],
        )

    def _draft(self, text: str) -> DraftReply:
        sender = _field(text, "From") or "unknown@example.com"
        subject = _field(text, "Subject") or ""
        sign_off = _field(text, "Preference sign_off") or "Best regards,\nErland"
        short = _field(text, "Preference reply_length") == "short"
        facts = re.findall(r"^\[(F\d+)\]\s*(.+)$", text, flags=re.MULTILINE)

        name = sender.split("@")[0].split(".")[0].capitalize()
        lines = [f"Hi {name},", "", "Thanks for your email."]
        if facts and not short:
            lines.append(facts[0][1])
        lines += ["", sign_off]
        return DraftReply(
            to=[sender],
            subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
            body="\n".join(lines),
            tone="concise" if short else "friendly",
            facts_used=[facts[0][0]] if facts and not short else [],
        )

    def _schedule(self, text: str) -> ScheduleProposal:
        subject = _field(text, "Subject") or "Follow-up"
        sender = _field(text, "From")
        slot = _field(text, "Free slot")
        if slot:
            start, end = (datetime.fromisoformat(part.strip()) for part in slot.split("|"))
            return ScheduleProposal(
                kind="event",
                title=subject.removeprefix("Re: "),
                start=start,
                end=end,
                attendees=[sender] if sender else [],
                reason="First free slot in working hours.",
            )
        due = _field(text, "Due")
        if not due:
            raise LLMError("FakeLLM needs a 'Free slot:' or 'Due:' line to schedule")
        return ScheduleProposal(
            kind="task",
            title=f"Follow up: {subject.removeprefix('Re: ')}",
            due=datetime.fromisoformat(due),
            reason="Sender said they would come back; remind to follow up.",
        )
