# betsson-email-copilot

A single-user productivity copilot on top of one existing mailbox and calendar, not a mail system.

It is an agent system in Python and LangGraph that triages, summarises, drafts and schedules, with human approval before every outgoing action, memory, and OpenTelemetry tracing. Nothing is ever really sent: mail and calendar go through mock adapters backed by JSON files.

> Work in progress, built in phases (see `PLAN.md`, section 11). Phase 1 is done: config, schemas, mock adapters, sample data and the LLM client.

## Quick start (no API keys needed)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                           # optional; LLM_PROVIDER=fake is the default
.venv/bin/pytest
```

Requires Python 3.11 or newer.

## Layout so far

| Path | What it is |
|---|---|
| `src/assistant/config.py` | Settings loaded from `.env` and validated by Pydantic |
| `src/assistant/schemas.py` | Every message between agents, tools and the human |
| `src/assistant/adapters/` | `MailAdapter` and `CalendarAdapter` interfaces plus JSON-file mocks |
| `src/assistant/llm/` | `LLMClient` interface, deterministic `FakeLLM`, stubs for OpenAI and Anthropic |
| `data/` | Sample inbox, calendar and organisational facts |
| `DECISIONS.md` | Design decisions with the alternatives and trade-offs |
