"""OpenTelemetry tracing: one trace per email run, spans per node, LLM call and tool call.

Spans always go to logs/traces.jsonl (one JSON object per line), which
scripts/show_trace.py turns into a waterfall. When OTEL_EXPORTER_OTLP_ENDPOINT
is set they also go to an OTLP collector such as Jaeger.

The provider is not registered globally: it lives on Deps like everything
else, so tests get their own and nothing leaks between runs.
"""

from pathlib import Path
from typing import IO

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

SERVICE_NAME = "email-copilot"


def _one_line(span: ReadableSpan) -> str:
    return span.to_json(indent=None) + "\n"


def make_tracer_provider(traces_file: IO[str], otlp_endpoint: str | None = None) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    # The console exporter writes to any file; Simple (not Batch) so each span is on disk
    # as soon as it ends, even if the process is stopped at an approval prompt.
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=traces_file, formatter=_one_line)))
    if otlp_endpoint:
        # Imported only when used. An endpoint passed in code must include the signal path.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return provider


def open_traces_file(logs_dir: Path) -> IO[str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    return (logs_dir / "traces.jsonl").open("a", encoding="utf-8")
