"""Logs, traces and metrics, bundled so the rest of the code needs one object.

    obs.log      structlog logger -> logs/app.jsonl (errors also -> logs/errors.jsonl)
    obs.tracer   OpenTelemetry tracer -> logs/traces.jsonl (+ OTLP if configured)
    obs.metrics  counters and histograms -> logs/metrics.json on close()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import structlog
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from assistant.observability.logging import make_logger
from assistant.observability.metrics import Metrics
from assistant.observability.tracing import make_tracer_provider, open_traces_file


@dataclass
class Observability:
    log: structlog.BoundLogger
    tracer: Tracer
    metrics: Metrics
    logs_dir: Path
    _provider: TracerProvider
    _files: list[IO[str]]
    _closed: bool = field(default=False)

    @classmethod
    def open(cls, logs_dir: Path, otlp_endpoint: str | None = None) -> "Observability":
        logs_dir.mkdir(parents=True, exist_ok=True)
        app_file = (logs_dir / "app.jsonl").open("a", encoding="utf-8")
        errors_file = (logs_dir / "errors.jsonl").open("a", encoding="utf-8")
        traces_file = open_traces_file(logs_dir)
        provider = make_tracer_provider(traces_file, otlp_endpoint)
        return cls(
            log=make_logger(app_file, errors_file),
            tracer=provider.get_tracer("assistant"),
            metrics=Metrics(),
            logs_dir=logs_dir,
            _provider=provider,
            _files=[app_file, errors_file, traces_file],
        )

    def close(self) -> None:
        """Flush spans, write metrics.json and close the log files. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._provider.shutdown()
        self.metrics.save(self.logs_dir / "metrics.json")
        for f in self._files:
            f.close()


__all__ = ["Metrics", "Observability"]
