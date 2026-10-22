"""Turn logs/traces.jsonl into a terminal waterfall, one trace at a time.

    email_run                        |##############################| 14.2 ms
      node triage                    | ###                          |  1.6 ms
        chat fake-triage             | ##                           |  1.1 ms  112 in / 30 out
"""

import json
from datetime import datetime
from pathlib import Path

BAR = 30


def load_spans(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trace_ids(spans: list[dict]) -> list[str]:
    """Traces in the order their root span ended, oldest first."""
    return [s["context"]["trace_id"] for s in spans if s["parent_id"] is None]


def find_trace(spans: list[dict], email_id: str | None = None) -> str | None:
    """The latest trace, or the latest one for this email."""
    roots = [s for s in spans if s["parent_id"] is None]
    if email_id:
        roots = [s for s in roots if s["attributes"].get("assistant.email_id") == email_id]
    return roots[-1]["context"]["trace_id"] if roots else None


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _details(span: dict) -> str:
    a = span["attributes"]
    parts = []
    if "gen_ai.usage.input_tokens" in a:
        parts.append(f"{a['gen_ai.usage.input_tokens']} in / {a['gen_ai.usage.output_tokens']} out")
    if a.get("assistant.waiting_for_human"):
        parts.append("waiting for human")
    if span["status"]["status_code"] == "ERROR":
        parts.append(f"ERROR {span['status'].get('description') or ''}".strip())
    return "  ".join(parts)


def render(spans: list[dict], trace_id: str) -> str:
    spans = [s for s in spans if s["context"]["trace_id"] == trace_id]
    root = next(s for s in spans if s["parent_id"] is None)
    t0, t1 = _time(root["start_time"]), _time(root["end_time"])
    total = max((t1 - t0).total_seconds(), 1e-9)
    children: dict[str, list[dict]] = {}
    for s in spans:
        children.setdefault(s["parent_id"], []).append(s)

    a = root["attributes"]
    lines = [f"Trace {trace_id.removeprefix('0x')}  email {a.get('assistant.email_id', '?')}  "
             f"run {a.get('assistant.run_id', '?')}  {total * 1000:.1f} ms"]

    def walk(span: dict, depth: int) -> None:
        start, end = _time(span["start_time"]), _time(span["end_time"])
        offset = round((start - t0).total_seconds() / total * BAR)
        width = max(1, round((end - start).total_seconds() / total * BAR))
        bar = (" " * offset + "#" * width).ljust(BAR)[:BAR]
        name = ("  " * depth + span["name"])[:40].ljust(40)
        ms = (end - start).total_seconds() * 1000
        lines.append(f"{name} |{bar}| {ms:7.1f} ms  {_details(span)}".rstrip())
        for child in sorted(children.get(span["context"]["span_id"], []), key=lambda s: s["start_time"]):
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines)
