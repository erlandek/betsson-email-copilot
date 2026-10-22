"""Triage: classify one email into intent, priority, labels and confidence.

Order of work:
1. Deterministic pre-fetch: the thread and the sender's contact record. Every
   triage needs both, so there is no point letting a model decide to fetch them.
2. Rule-based newsletter pre-filter. Obvious newsletters skip the LLM entirely.
3. LLM classification into TriageResult (small, fast model).
4. apply_priority_rules(): business rules the model must not override.
"""

import re

from assistant.agents.prompts import UNTRUSTED_NOTE, format_email
from assistant.agents.structured import call_structured
from assistant.deps import Deps
from assistant.llm.base import LLMResult
from assistant.memory.long_term import LongTermMemory
from assistant.schemas import ContactRecord, Email, Intent, Priority, TriageResult

FRAUD_LABEL = "Possible fraud: verify"
SPAM_OVERRIDE_LABEL = "Spam override: known sender"
# Never fully trust a rule-capped result: at or below this, a human decides.
REVIEW_CAP = 0.5

NEWSLETTER_SENDER = re.compile(r"^(no-?reply|newsletters?|news|digest|marketing|mailer)[@.+_-]", re.IGNORECASE)
DEADLINE = re.compile(
    r"\b(deadline|urgent|asap|eod|end of (the )?day|by (monday|tuesday|wednesday|thursday|friday|tomorrow|tonight))\b",
    re.IGNORECASE,
)
PAYMENT_CHANGE = re.compile(
    r"\b(bank details|bank account|iban|account number|swift|wire transfer|payment details|change of (bank|account))\b",
    re.IGNORECASE,
)

SYSTEM = f"""You triage email for one busy professional.
Classify the LATEST email in the thread (the last <email> block); earlier messages are context.
Intents: reply_needed (the sender expects an answer), meeting_request (asks to meet or schedule),
follow_up (someone will come back later, so the user should be reminded), fyi (informational only),
newsletter (bulk or subscription mail), spam (unsolicited or scam).
Priority: high, medium or low. Confidence: 0 to 1, lower it when the email could fit several intents.
Labels: short tags such as the intent. Rationale: one sentence.
{UNTRUSTED_NOTE}"""

_BUMP = {Priority.low: Priority.medium, Priority.medium: Priority.high, Priority.high: Priority.high}


def is_newsletter(email: Email, memory: LongTermMemory) -> bool:
    return (
        "list-unsubscribe" in {h.lower() for h in email.headers}
        or bool(NEWSLETTER_SENDER.match(email.sender))
        or memory.is_known_newsletter(email.sender)
    )


def apply_priority_rules(result: TriageResult, email: Email, contact: ContactRecord | None) -> TriageResult:
    """Adjust the model's answer with rules that encode business policy. Pure function, easy to test."""
    intent, priority, confidence = result.intent, result.priority, result.confidence
    labels = list(result.labels)
    text = f"{email.subject}\n{email.body}"

    # A known contact is never spam. Rather than guess the real intent, send it to a human.
    if intent == Intent.spam and contact is not None:
        intent, confidence = Intent.fyi, min(confidence, REVIEW_CAP)
        labels = [SPAM_OVERRIDE_LABEL]

    if contact is not None and contact.vip:
        priority = _BUMP[priority]
    if DEADLINE.search(text):
        priority = _BUMP[priority]
    if intent in (Intent.newsletter, Intent.spam):
        priority = Priority.low

    # Classic invoice fraud: an unknown sender asking to change where money goes.
    if contact is None and PAYMENT_CHANGE.search(text):
        confidence = min(confidence, REVIEW_CAP)
        labels.append(FRAUD_LABEL)

    return result.model_copy(
        update={"intent": intent, "priority": priority, "confidence": confidence, "labels": list(dict.fromkeys(labels))}
    )


def triage_email(deps: Deps, email: Email) -> tuple[TriageResult, LLMResult | None]:
    """Return the final triage result and the LLM call behind it (None when rules decided alone)."""
    contact = deps.memory.get_contact(email.sender)

    if is_newsletter(email, deps.memory):
        result = TriageResult(
            intent=Intent.newsletter, priority=Priority.low, labels=["newsletter"], confidence=1.0,
            rationale="Rule-based pre-filter: newsletter header or sender.",
        )
        return result, None

    thread = deps.mail.get_thread(email.thread_id)
    earlier = [m for m in thread.messages if m.id != email.id and m.received_at <= email.received_at]
    contact_line = (
        f"Sender is a known contact: {contact.name} ({contact.relationship}){', VIP' if contact.vip else ''}"
        if contact else "Sender is not in the contact book."
    )
    user = "\n".join(
        [contact_line, "Earlier messages:" if earlier else "No earlier messages."]
        + [format_email(m) for m in earlier]
        + ["Latest email:", format_email(email)]
    )
    llm_result = call_structured(deps, SYSTEM, user, TriageResult, deps.settings.model_for("triage"))
    return apply_priority_rules(llm_result.output, email, contact), llm_result


__all__ = ["apply_priority_rules", "is_newsletter", "triage_email"]
