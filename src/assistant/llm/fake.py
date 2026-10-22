"""A deterministic stand-in for a real LLM.

FakeLLM lets the whole demo and test suite run with no API keys and identical
output on every run. It does not try to be clever: it picks a small handler per
output schema and applies keyword rules to the prompt text.

Prompt convention the handlers rely on. The agents build prompts this way
anyway, and tool results are rendered in the same line format:
    From: <address>          Subject: <subject>          Received: <iso>
    Body: ... (until the next From: line or </email>)
    [F02] <org fact text>    Preference sign_off: <text>
    Search window: <iso start> | <iso end>
    Free time: <iso start> | <iso end>    Due: <iso datetime>

In a tool loop, FakeLLM follows a fixed script per output schema (which tools,
in which order, with which arguments), then answers from the transcript. A real
model chooses its own calls; the loop enforces the same limits on both.
"""

import re
from datetime import datetime, timedelta

from assistant.llm.base import LLMClient, LLMError, LLMResult, Message, T, ToolSpec
from assistant.schemas import (
    DraftReply,
    Intent,
    LearnedPreferences,
    PreferenceUpdate,
    Priority,
    ScheduleProposal,
    ThreadSummary,
    ToolCall,
    TriageResult,
)

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

# Topic tags, like the folders or labels a person would use. They describe what
# an email is about, which the intent already does not cover.
TAG_KEYWORDS: dict[str, list[str]] = {
    "Finance": ["invoice", "iban", "bank details", "payment run", "payment details"],
    "Legal": ["agreement", "contract", "legal"],
    "Reporting": ["kpi", "board pack", "numbers"],
    "Partners": ["integration", "sandbox", "partner", "api "],
    "IT": ["network", "maintenance", "vpn"],
    "Meetings": ["invite", "30 minutes", "set up", "sync"],
    "Promotions": ["prize", "offer", "reward"],
}
INTERNAL_DOMAIN = "example.com"


def _tags(email_text: str) -> list[str]:
    lowered = email_text.lower()
    tags = [tag for tag, words in TAG_KEYWORDS.items() if any(w in lowered for w in words)]
    sender = _field(email_text, "From") or ""
    if sender.lower().endswith("@" + INTERNAL_DOMAIN):
        tags.insert(0, "Internal")
    return tags[:3]


def _field(text: str, name: str) -> str | None:
    match = re.search(rf"^{name}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _bodies(text: str) -> list[str]:
    # A body runs from "Body:" until the next message's "From:" line, a closing tag or the end.
    pattern = r"^Body:\s*(.*?)(?=^From:|^</email>|\Z)"
    return [b.strip() for b in re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)]


def _latest_field(text: str, name: str) -> str | None:
    """Like _field, but the last occurrence: in a thread, the latest message counts."""
    matches = re.findall(rf"^{name}:\s*(.+)$", text, flags=re.MULTILINE)
    return matches[-1].strip() if matches else None


def _tool_script(schema: type, text: str) -> list[ToolCall]:
    """The fixed sequence of tool calls FakeLLM makes for each output schema."""
    if schema is DraftReply:
        sender = _latest_field(text, "From") or ""
        subject = _latest_field(text, "Subject") or ""
        bodies = _bodies(text)
        query = f"{subject} {' '.join(bodies[-1].split()[:40]) if bodies else ''}".strip()
        return [
            ToolCall(name="lookup_contact", args={"email": sender}),
            ToolCall(name="search_knowledge", args={"query": query, "k": 3}),
            ToolCall(name="get_preferences", args={}),
        ]
    if schema is ScheduleProposal:
        calls = [ToolCall(name="get_preferences", args={})]
        window = _field(text, "Search window")
        if window:
            start, end = (part.strip() for part in window.split("|"))
            calls.append(ToolCall(name="check_availability", args={"start": start, "end": end, "duration_minutes": 30}))
        return calls
    return []


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
            LearnedPreferences: self._learn,
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

    def next_step(
        self, system: str, messages: list[Message], tools: list[ToolSpec], schema: type[T], model: str
    ) -> ToolCall | LLMResult[T]:
        transcript = "\n".join(m.content for m in messages)
        calls_made = sum(1 for m in messages if m.role == "tool_call")
        script = _tool_script(schema, messages[0].content if messages else "")
        offered = {t.name for t in tools}
        if calls_made < len(script) and script[calls_made].name in offered:
            return script[calls_made]
        return self.generate(system, transcript, schema, model)

    def _triage(self, text: str) -> TriageResult:
        # Classify only the latest email; earlier messages are context for a real model.
        blocks = re.findall(r"<email>(.*?)</email>", text, flags=re.DOTALL)
        latest = blocks[-1] if blocks else text
        lowered = latest.lower()
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
            labels=_tags(latest),
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
        sender = _latest_field(text, "From") or "unknown@example.com"
        subject = _latest_field(text, "Subject") or ""
        sign_off = _field(text, "Preference sign_off") or "Best regards, Erland"
        short = _field(text, "Preference reply_length") == "short"
        # A short reply cites one fact, a normal reply up to three; duplicates are dropped.
        facts = list(dict.fromkeys(re.findall(r"^\[(F\d+)\]\s*(.+)$", text, flags=re.MULTILINE)))
        facts = facts[:1] if short else facts[:3]

        name = sender.split("@")[0].split(".")[0].split("-")[0].capitalize()
        lines = [f"Hi {name},", "", "Thanks for your email." if not short else "Thanks."]
        lines += [fact for _, fact in facts]
        lines += ["", sign_off]
        return DraftReply(
            to=[sender],
            subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
            body="\n".join(lines),
            tone="concise" if short else "friendly",
            facts_used=[fact_id for fact_id, _ in facts],
        )

    def _learn(self, text: str) -> LearnedPreferences:
        """Two simple diff rules: a new last line is a sign-off, a much shorter draft means short replies."""
        before = re.search(r"Draft before:\n<<<\n(.*?)\n>>>", text, flags=re.DOTALL)
        after = re.search(r"Draft after:\n<<<\n(.*?)\n>>>", text, flags=re.DOTALL)
        if not before or not after:
            return LearnedPreferences()
        before_text, after_text = before.group(1).strip(), after.group(1).strip()
        updates = []
        last_before, last_after = before_text.splitlines()[-1].strip(), after_text.splitlines()[-1].strip()
        if last_after and last_after != last_before and len(last_after) <= 40:
            updates.append(PreferenceUpdate(key="sign_off", value=last_after))
        if len(after_text) < 0.7 * len(before_text):
            updates.append(PreferenceUpdate(key="reply_length", value="short"))
        return LearnedPreferences(updates=updates)

    def _schedule(self, text: str) -> ScheduleProposal:
        subject = _latest_field(text, "Subject") or "Follow-up"
        sender = _latest_field(text, "From")
        periods = [
            tuple(datetime.fromisoformat(part.strip()) for part in line.split("|"))
            for line in re.findall(r"^Free time:\s*(.+)$", text, flags=re.MULTILINE)
        ]
        # Honour weekdays and "afternoon" named in the request, when such time is free.
        bodies = _bodies(text)
        request = bodies[-1].lower() if bodies else ""
        days = [d for d in ("monday", "tuesday", "wednesday", "thursday", "friday") if d in request]
        if days:
            periods = [p for p in periods if p[0].strftime("%A").lower() in days] or periods
        if "afternoon" in request:
            # Start at 13:00 when a free period spans lunchtime.
            afternoon = [(max(s, s.replace(hour=13, minute=0)), e) for s, e in periods if e.hour >= 14]
            periods = afternoon or periods
        if periods:
            start = periods[0][0]
            minutes = int(_field(text, "Preference meeting_length_minutes") or 30)
            return ScheduleProposal(
                kind="event",
                title=subject.removeprefix("Re: "),
                start=start,
                end=start + timedelta(minutes=minutes),
                attendees=[sender] if sender else [],
                reason="First free time in working hours that matches the request.",
            )
        due = _field(text, "Due")
        received = _latest_field(text, "Received")
        if due:
            due_at = datetime.fromisoformat(due)
        elif received:
            # No explicit date: remind a week after the email, at 09:00.
            due_at = (datetime.fromisoformat(received) + timedelta(days=7)).replace(hour=9, minute=0, second=0)
        else:
            raise LLMError("FakeLLM needs a 'Free slot:', 'Due:' or 'Received:' line to schedule")
        return ScheduleProposal(
            kind="task",
            title=f"Follow up: {subject.removeprefix('Re: ')}",
            due=due_at,
            reason="Sender said they would come back; remind to follow up.",
        )
