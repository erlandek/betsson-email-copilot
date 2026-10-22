# betsson-email-copilot

A single-user productivity copilot on top of one existing mailbox and calendar, not a mail system.

It is an agent system in Python and LangGraph that triages, summarises, drafts and schedules, with human approval before every outgoing action, memory, and OpenTelemetry tracing. Nothing is ever really sent: mail and calendar go through mock adapters backed by JSON files.

> Work in progress, built in phases (see `PLAN.md`, section 11). Done so far: Phase 1 (config, schemas, mock adapters, sample data, LLM client) and Phase 2 (triage, summariser, drafter and scheduler agents, the LangGraph orchestrator and router). Next: human approval, execution and memory.

## Quick start (no API keys needed)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                           # optional; LLM_PROVIDER=fake is the default
.venv/bin/pytest
.venv/bin/python scripts/process_inbox.py      # run the inbox through the graph and print a triage table
```

Requires Python 3.11 or newer.

## Layout so far

| Path | What it is |
|---|---|
| `src/assistant/config.py` | Settings loaded from `.env` and validated by Pydantic |
| `src/assistant/schemas.py` | Every message between agents, tools and the human |
| `src/assistant/adapters/` | `MailAdapter` and `CalendarAdapter` interfaces plus JSON-file mocks |
| `src/assistant/llm/` | `LLMClient` interface, deterministic `FakeLLM`, stubs for OpenAI and Anthropic |
| `src/assistant/agents/` | Triage (pre-filter, rules), summariser, drafter, scheduler, and the bounded tool loop |
| `src/assistant/tools.py` | Read-only tools with validated arguments and per-agent allowlists |
| `src/assistant/graph.py` | The LangGraph orchestrator and deterministic router |
| `src/assistant/policy.py` | `requires_approval()`, driven by `APPROVAL_MODE` |
| `src/assistant/memory/` | Contacts, preferences and org facts (read-only until Phase 3) |
| `data/` | Sample inbox, calendar and organisational facts |
| `DECISIONS.md` | Design decisions with the alternatives and trade-offs |
