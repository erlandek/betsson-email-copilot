"""Structured JSON logs with structlog.

Every line in logs/app.jsonl carries the run_id, email_id and the current
trace_id and span_id, so a log line can be matched to its trace. Errors are
also copied to logs/errors.jsonl, so failures can be found without scanning
everything.
"""

import json
from typing import IO, Any

import structlog
from opentelemetry import trace


def add_trace_ids(logger: Any, method: str, event: dict[str, Any]) -> dict[str, Any]:
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event["trace_id"] = format(context.trace_id, "032x")
        event["span_id"] = format(context.span_id, "016x")
    return event


def copy_errors_to(errors_file: IO[str]):
    def copy(logger: Any, method: str, event: dict[str, Any]) -> dict[str, Any]:
        if method in ("error", "exception", "critical"):
            errors_file.write(json.dumps(event, default=str) + "\n")
            errors_file.flush()
        return event

    return copy


def make_logger(app_file: IO[str], errors_file: IO[str]) -> structlog.BoundLogger:
    # wrap_logger (not structlog.configure) keeps the logger on Deps instead of in global config.
    return structlog.wrap_logger(
        structlog.WriteLogger(app_file),
        processors=[
            structlog.contextvars.merge_contextvars,  # run_id and email_id, bound per run
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            add_trace_ids,
            copy_errors_to(errors_file),
            structlog.processors.JSONRenderer(default=str),
        ],
    )
