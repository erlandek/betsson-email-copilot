"""Evaluate the triage classifier on data/eval_emails.json, for one or more models.

    python scripts/eval_classifier.py                                    # FakeLLM only, no keys
    python scripts/eval_classifier.py fake openai:gpt-6-luna anthropic:claude-haiku-5-5
    python scripts/eval_classifier.py openai:gpt-6-luna --markdown eval_results.md

Each model is given as provider:model (keys come from .env). Fallback is off, so
every number belongs to the model named. Prints markdown tables: a comparison
of all models, then per-intent precision and recall, a confusion matrix and
accuracy by confidence bucket for each.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant.evaluation import load_cases, report, run_model, summarise  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "eval_emails.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=["fake"], help="fake or provider:model")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--markdown", type=Path, help="also write the report to this file")
    args = parser.parse_args()

    cases = load_cases(args.data)
    results = {}
    for spec in args.models:
        print(f"Running {spec} on {len(cases)} emails...", file=sys.stderr)
        predictions = run_model(spec, cases)
        results[spec] = (predictions, summarise(predictions))
    text = report(results)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
