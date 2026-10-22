"""Print a trace from logs/traces.jsonl as a waterfall in the terminal.

    python scripts/show_trace.py            # the latest email run
    python scripts/show_trace.py e04        # the latest run for email e04
    python scripts/show_trace.py --all      # every run in the file
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant.observability.waterfall import find_trace, load_spans, render, trace_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("email_id", nargs="?", help="show the latest run for this email")
    parser.add_argument("--all", action="store_true", help="show every run")
    parser.add_argument("--file", type=Path, default=Path("logs/traces.jsonl"))
    args = parser.parse_args()

    if not args.file.exists():
        sys.exit(f"No traces yet: {args.file} does not exist. Run scripts/process_inbox.py first.")
    spans = load_spans(args.file)
    ids = trace_ids(spans) if args.all else [find_trace(spans, args.email_id)]
    if not ids or ids == [None]:
        sys.exit(f"No trace found{' for ' + args.email_id if args.email_id else ''}.")
    print("\n\n".join(render(spans, trace_id) for trace_id in ids))


if __name__ == "__main__":
    main()
