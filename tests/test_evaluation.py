"""The classifier evaluation (DECISIONS.md, D38)."""

from conftest import DATA_DIR

from assistant.evaluation import Prediction, load_cases, report, run_model, summarise
from assistant.schemas import Intent

EVAL = DATA_DIR / "eval_emails.json"


def test_eval_set_covers_every_intent_with_unique_ids():
    cases = load_cases(EVAL)
    assert len(cases) == 20
    assert {c.expected_intent for c in cases} == {i.value for i in Intent}
    assert len({c.email.id for c in cases}) == len(cases)
    assert all(c.why for c in cases)


def test_fake_run_is_deterministic_and_prefilter_costs_nothing():
    cases = load_cases(EVAL)
    first, second = run_model("fake", cases), run_model("fake", cases)
    assert [(p.intent, p.confidence) for p in first] == [(p.intent, p.confidence) for p in second]
    newsletter = next(p for p in first if p.id == "v16")  # has a List-Unsubscribe header
    assert newsletter.intent == "newsletter" and newsletter.input_tokens == 0


def p(expected, got, confidence=0.9, review=False):
    return Prediction(id="x", expected_intent=expected, expected_priority="low", intent=got, priority="low",
                      confidence=confidence, review=review)


def test_summary_metrics():
    preds = [
        p("reply_needed", "reply_needed"),
        p("reply_needed", "meeting_request", review=False),  # wrong and acted on
        p("meeting_request", "meeting_request", confidence=0.5, review=True),
        p("spam", None, confidence=0.0, review=True),  # failed twice
    ]
    s = summarise(preds)
    assert s["intent_accuracy"] == 0.5
    assert s["per_intent"]["reply_needed"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert s["per_intent"]["meeting_request"]["precision"] == 0.5
    assert s["confusion"]["reply_needed"]["meeting_request"] == 1 and s["confusion"]["spam"][None] == 1
    assert s["silent_errors"] == 1 and s["failures"] == 0  # failures count AgentErrors, set separately
    review_bucket = s["confidence_buckets"][0]
    assert review_bucket["count"] == 1 and review_bucket["accuracy"] == 1.0


def test_report_compares_models_in_one_table():
    preds = [p("fyi", "fyi")]
    text = report({"fake": (preds, summarise(preds)), "other": (preds, summarise(preds))})
    table = text.split("###")[0]
    assert "| fake |" in table and "| other |" in table and "Confusion matrix" in text
