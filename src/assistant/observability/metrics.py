"""A small in-process metrics registry, saved to logs/metrics.json after a run.

Counters and histograms keyed by name plus labels, e.g.
tool_calls_total{outcome=ok,tool=search_knowledge}. Enough to show the approach
without running a monitoring stack; the same names would map one to one onto
Prometheus or OpenTelemetry metrics in production.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def _key(name: str, labels: dict[str, object]) -> str:
    if not labels:
        return name
    inner = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1, **labels: object) -> None:
        self.counters[_key(name, labels)] += value

    def observe(self, name: str, value: float, **labels: object) -> None:
        self.histograms[_key(name, labels)].append(value)

    def snapshot(self) -> dict:
        return {
            "counters": dict(sorted(self.counters.items())),
            "histograms": {key: _summary(values) for key, values in sorted(self.histograms.items())},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        """Plain-text summary printed after the demo."""
        snap = self.snapshot()
        lines = ["Metrics"]
        lines += [f"  {key:<58} {value:g}" for key, value in snap["counters"].items()]
        for key, h in snap["histograms"].items():
            lines.append(f"  {key:<58} n={h['count']} avg={h['avg']:.1f} p95={h['p95']:.1f} max={h['max']:.1f}")
        return "\n".join(lines)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": p95,
        "max": ordered[-1],
    }
