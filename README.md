# betsson-email-copilot

A single-user productivity copilot on top of one existing mailbox and calendar, not a mail system.

It is an agent system in Python and LangGraph that triages, summarises, drafts and schedules, with human approval before every outgoing action, memory, and OpenTelemetry tracing. Nothing is ever really sent: mail and calendar go through mock adapters backed by JSON files.

> Work in progress, built in phases (see `PLAN.md`, section 11). Done so far: Phase 1 (config, schemas, mock adapters, sample data, LLM client), Phase 2 (agents, LangGraph orchestrator, router) and Phase 3 (human approval, exactly-once execution, SQLite and Chroma memory, learning from edits), real OpenAI and Anthropic providers, and Phase 4 (logs, traces, metrics, retries and provider fallback).

## Quick start (no API keys needed)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                           # optional; LLM_PROVIDER=fake is the default
.venv/bin/pytest
.venv/bin/python scripts/process_inbox.py      # triage table and metrics; stops before any approval
.venv/bin/python scripts/show_trace.py e04     # waterfall of the latest run for an email
.venv/bin/python scripts/copilot.py run        # interactive: approve, edit, reject or skip each action
.venv/bin/python scripts/copilot.py memory     # preferences (seeded or learned) and feedback
.venv/bin/python scripts/copilot.py outbox     # what was "sent" and scheduled (mock files in runtime/)
```

## Running with a real model

Put a key in `.env` and pick the provider, either in `.env` or for one run:

```bash
LLM_PROVIDER=openai    .venv/bin/python scripts/process_inbox.py   # needs OPENAI_API_KEY
LLM_PROVIDER=anthropic .venv/bin/python scripts/process_inbox.py   # needs ANTHROPIC_API_KEY
```

On Windows PowerShell: `$env:LLM_PROVIDER="openai"; .venv\Scripts\python scripts\process_inbox.py`. Model names come from `.env` (a small model for triage, a stronger one for drafting). Output from real models is not deterministic, so the table can differ from the fake run.

If both keys are set, a call that still fails after the SDK's retries (rate limit, server error, timeout) moves to the other provider, using that provider's model for the same task. Set `LLM_FALLBACK=false` to turn this off.

## Observability

Every run writes to `logs/`:

| File | What |
|---|---|
| `app.jsonl` | JSON log lines with `run_id`, `email_id`, `trace_id`, node, tool, latency, tokens, cost |
| `errors.jsonl` | Only the errors, with the same ids, so a failure can be traced |
| `traces.jsonl` | OpenTelemetry spans: one trace per email run, a span per node, LLM call and tool call |
| `metrics.json` | Counters and latency histograms from the last run (also printed after the run) |

`scripts/show_trace.py [email_id]` prints a run as a waterfall. Set `OTEL_EXPORTER_OTLP_ENDPOINT` (for example `http://localhost:4318`) to send the same spans to Jaeger. Prompts and email bodies are not written to traces or logs (an error message can quote part of a bad model answer).

Delete `runtime/` to start the demo from scratch. The first run downloads Chroma's local embedding model (about 80 MB); set `KNOWLEDGE_BACKEND=keyword` to skip it.

Requires Python 3.11 or newer.

## Layout so far

| Path | What it is |
|---|---|
| `src/assistant/config.py` | Settings loaded from `.env` and validated by Pydantic |
| `src/assistant/schemas.py` | Every message between agents, tools and the human |
| `src/assistant/adapters/` | `MailAdapter` and `CalendarAdapter` interfaces plus JSON-file mocks |
| `src/assistant/llm/` | `LLMClient` interface, deterministic `FakeLLM`, OpenAI and Anthropic providers, price table, tracing wrapper, fallback |
| `src/assistant/agents/` | Triage (pre-filter, rules), summariser, drafter, scheduler, and the bounded tool loop |
| `src/assistant/tools.py` | Read-only tools with validated arguments and per-agent allowlists |
| `src/assistant/graph.py` | The LangGraph orchestrator and deterministic router |
| `src/assistant/policy.py` | `requires_approval()`, driven by `APPROVAL_MODE` |
| `src/assistant/memory/` | SQLite long-term memory (preferences, contacts, feedback) and Chroma org facts |
| `src/assistant/approval.py` | Approval view, edit validation, deterministic action ids |
| `src/assistant/checkpoints.py` | SQLite checkpointer so runs pause and resume |
| `src/assistant/observability/` | Logging (structlog), tracing (OpenTelemetry), metrics, trace waterfall |
| `src/assistant/cli.py` | `run`, `memory` and `outbox` commands |
| `data/` | Sample inbox, calendar and organisational facts |
| `DECISIONS.md` | Design decisions with the alternatives and trade-offs |
