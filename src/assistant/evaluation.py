"""Evaluate the triage classifier on a labelled set (PLAN section 6, amendment A10).

What is measured is the whole triage step as it runs in the graph: the
newsletter pre-filter, the model, and the priority and fraud rules. Each email
is its own one-message thread, so results do not depend on the sample inbox.

The labelled set lives in data/eval_emails.json. Each entry has the email
fields that matter (id, sender, subject, body, optional headers) plus
expected_intent, expected_priority and a one-line "why" for the label.
"""

import json
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from assistant.adapters import MockMailAdapter
from assistant.agents.structured import AgentError
from assistant.agents.triage import FRAUD_LABEL, LOOKALIKE_LABEL, triage_email
from assistant.config import Settings
from assistant.deps import build_deps
from assistant.schemas import Email, Intent

START = datetime.fromisoformat("2026-10-21T08:00:00+02:00")
INTENTS = [i.value for i in Intent]
# Confidence buckets; the lowest one is what the router sends to human review.
BUCKETS = [(0.0, 0.6, "< 0.6 (review)"), (0.6, 0.8, "0.6 to 0.8"), (0.8, 1.01, ">= 0.8")]


class Case(BaseModel):
    email: Email
    expected_intent: str
    expected_priority: str
    why: str = ""


class Prediction(BaseModel):
    id: str
    expected_intent: str
    expected_priority: str
    intent: str | None = None  # None when the model failed twice (the email would go to review)
    priority: str | None = None
    confidence: float = 0.0
    review: bool = True  # would the router send it to a human?
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def load_cases(path: Path, user_email: str = "erland@example.com") -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for i, item in enumerate(raw):
        email = Email(
            id=item["id"], thread_id=f"eval-{item['id']}", sender=item["sender"], recipients=[user_email],
            subject=item["subject"], body=item["body"], headers=item.get("headers", {}),
            received_at=START + timedelta(minutes=10 * i),
        )
        cases.append(Case(email=email, expected_intent=item["expected_intent"],
                          expected_priority=item["expected_priority"], why=item.get("why", "")))
    return cases


def settings_for(spec: str, workdir: Path) -> Settings:
    """'fake' or 'provider:model', e.g. 'openai:gpt-5.4-mini'. Keys still come from .env."""
    provider, _, model = spec.partition(":")
    overrides = {"llm_provider": provider, "llm_fallback": False,  # measure the model asked for, not a backup
                 "knowledge_backend": "keyword", "runtime_dir": workdir / "runtime", "logs_dir": workdir / "logs"}
    if model:
        overrides[f"{provider}_triage_model"] = model
    return Settings(**overrides)


def run_model(spec: str, cases: list[Case]) -> list[Prediction]:
    workdir = Path(tempfile.mkdtemp(prefix="copilot-eval-"))
    settings = settings_for(spec, workdir)
    deps = build_deps(settings)
    inbox = workdir / "eval_inbox.json"
    inbox.write_text(json.dumps([c.email.model_dump(mode="json") for c in cases]), encoding="utf-8")
    deps.mail = MockMailAdapter(inbox, workdir / "sent.json")

    predictions = []
    try:
        for case in cases:
            p = Prediction(id=case.email.id, expected_intent=case.expected_intent,
                           expected_priority=case.expected_priority)
            started = time.perf_counter()
            try:
                result, llm = triage_email(deps, case.email)
            except AgentError as error:
                p.error = str(error)[:200]
            else:
                p.intent, p.priority, p.confidence = result.intent.value, result.priority.value, result.confidence
                warned = FRAUD_LABEL in result.labels or LOOKALIKE_LABEL in result.labels
                p.review = warned or result.confidence < settings.confidence_threshold
                if llm:
                    p.cost_usd, p.input_tokens, p.output_tokens = llm.cost_usd, llm.input_tokens, llm.output_tokens
            p.latency_ms = (time.perf_counter() - started) * 1000
            predictions.append(p)
    finally:
        deps.obs.close()
    return predictions


def summarise(preds: list[Prediction]) -> dict:
    n = len(preds)
    latencies = sorted(p.latency_ms for p in preds)
    per_intent = {}
    for intent in INTENTS:
        tp = sum(p.intent == intent and p.expected_intent == intent for p in preds)
        predicted = sum(p.intent == intent for p in preds)
        actual = sum(p.expected_intent == intent for p in preds)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent[intent] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    confusion = {a: {b: sum(p.expected_intent == a and p.intent == b for p in preds) for b in [*INTENTS, None]}
                 for a in INTENTS}
    buckets = []
    for low, high, name in BUCKETS:
        inside = [p for p in preds if p.intent and low <= p.confidence < high]
        correct = sum(p.intent == p.expected_intent for p in inside)
        buckets.append({"bucket": name, "count": len(inside), "accuracy": correct / len(inside) if inside else None})
    return {
        "emails": n,
        "intent_accuracy": sum(p.intent == p.expected_intent for p in preds) / n,
        "priority_accuracy": sum(p.priority == p.expected_priority for p in preds) / n,
        "macro_f1": statistics.fmean(v["f1"] for v in per_intent.values()),
        "review_rate": sum(p.review for p in preds) / n,
        # The costly mistakes: a wrong intent that the router would act on without a human.
        "silent_errors": sum(p.intent != p.expected_intent and not p.review for p in preds),
        "failures": sum(p.error is not None for p in preds),
        "latency_ms_avg": statistics.fmean(latencies),
        "latency_ms_p95": latencies[min(n - 1, round(0.95 * (n - 1)))],
        "cost_usd_per_email": sum(p.cost_usd for p in preds) / n,
        "tokens_per_email": sum(p.input_tokens + p.output_tokens for p in preds) / n,
        "per_intent": per_intent,
        "confusion": confusion,
        "confidence_buckets": buckets,
    }


# --- Report ------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def comparison_table(summaries: dict[str, dict]) -> str:
    rows = ["| Model | Intent acc. | Macro F1 | Priority acc. | To review | Wrong, not reviewed | Failures "
            "| Avg ms | p95 ms | Cost/email |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for spec, s in summaries.items():
        rows.append(
            f"| {spec} | {_pct(s['intent_accuracy'])} | {s['macro_f1']:.2f} | {_pct(s['priority_accuracy'])} | "
            f"{_pct(s['review_rate'])} | {s['silent_errors']} | {s['failures']} | "
            f"{s['latency_ms_avg']:.0f} | {s['latency_ms_p95']:.0f} | ${s['cost_usd_per_email']:.5f} |"
        )
    return "\n".join(rows)


def detail(spec: str, s: dict, preds: list[Prediction]) -> str:
    out = [f"### {spec}", "", "| Intent | Precision | Recall | F1 | Emails |", "|---|---|---|---|---|"]
    for intent, m in s["per_intent"].items():
        out.append(f"| {intent} | {_pct(m['precision'])} | {_pct(m['recall'])} | {m['f1']:.2f} | {m['support']} |")

    short = {i: i.replace("_", " ").split()[0] for i in INTENTS}  # "reply", "meeting", ... to keep it narrow
    out += ["", "Confusion matrix (rows: expected, columns: predicted; 'failed' = no valid answer)", "",
            "| expected \\ predicted | " + " | ".join(short.values()) + " | failed |",
            "|---" * (len(INTENTS) + 2) + "|"]
    for expected, row in s["confusion"].items():
        out.append(f"| {expected} | " + " | ".join(str(row[i]) for i in INTENTS) + f" | {row[None]} |")

    out += ["", "| Confidence | Emails | Intent accuracy |", "|---|---|---|"]
    out += [f"| {b['bucket']} | {b['count']} | {_pct(b['accuracy'])} |" for b in s["confidence_buckets"]]

    wrong = [p for p in preds if p.intent != p.expected_intent]
    if wrong:
        out += ["", "Misclassified: " + ", ".join(
            f"{p.id} ({p.expected_intent} -> {p.intent or 'failed'}, conf {p.confidence:.2f})" for p in wrong)]
    return "\n".join(out)


def report(results: dict[str, tuple[list[Prediction], dict]]) -> str:
    summaries = {spec: s for spec, (_, s) in results.items()}
    n = next(iter(summaries.values()))["emails"]
    parts = [f"## Classifier evaluation ({n} labelled emails)", "", comparison_table(summaries)]
    for spec, (preds, s) in results.items():
        parts += ["", detail(spec, s, preds)]
    return "\n".join(parts)
