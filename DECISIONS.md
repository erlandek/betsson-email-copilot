# Design decisions

One entry per decision: what we chose, the alternative, and the trade-off. Newest at the bottom.

## Phase 1: foundations

### D1. The plan's `email-assistant/` tree lives at the repository root
- **Choice:** `src/`, `data/`, `tests/` and the docs sit directly in the repo root instead of an extra `email-assistant/` folder.
- **Alternative:** nest everything under `email-assistant/` as drawn in PLAN.md section 9.
- **Trade-off:** one less directory level and simpler commands; nothing is lost because the repo holds only this project.

### D2. Settings are a Pydantic `BaseSettings` class read from `.env`
- **Choice:** `pydantic-settings` loads and validates every setting at startup (`config.py`).
- **Alternative:** `os.getenv` calls scattered through the code.
- **Trade-off:** one small extra dependency, but typos like `LLM_PROVIDER=opneai` fail immediately with a clear message, keys are `SecretStr` so they never appear in logs, and all settings are documented in one place.

### D3. Strict schemas: enums, Literals and `extra="forbid"`
- **Choice:** every schema rejects unknown fields; intents, priorities and decisions are closed sets.
- **Alternative:** lenient models that ignore extra fields.
- **Trade-off:** more validation failures from a sloppy model, but those are caught, retried once with the error, then escalated (PLAN section 4), rather than silently passing bad data through the graph.

### D4. `ActionProposal.payload` is a typed union, checked against `action_type`
- **Choice:** `payload: DraftReply | ScheduleProposal` with a validator that the payload fits the action (send_email needs a draft, create_event an event).
- **Alternative:** `payload: dict`.
- **Trade-off:** slightly more code, but a human edit is validated against exactly the same schema as the original proposal, and a mismatched action cannot be executed.

### D5. The mail adapter has no delete method
- **Choice:** `MailAdapter` offers label and archive only (amendment A4).
- **Alternative:** a delete method that policy forbids calling.
- **Trade-off:** none worth having: if the method does not exist, no prompt injection or bug can call it. A test asserts it is absent.

### D6. What the mocks persist
- **Choice:** the mock mailbox keeps labels and archive flags in memory and writes only sent mail to `sent.json`. The mock calendar writes every change to its JSON file. The sample files in `data/` are never modified; the demo works on copies in `runtime/`.
- **Alternative:** write every change back to `inbox.json`.
- **Trade-off:** labels are lost between runs, which is fine because triage re-applies them (labels behave like a set). Sends and calendar entries are what must survive a restart, for idempotency and for the demo.

### D7. Idempotency is enforced in the adapters as well as the execute node
- **Choice:** `send`, `create_event` and `create_task` take an `action_id` and return the existing record if that id was already used (amendment A7).
- **Alternative:** rely only on the execute node checking first.
- **Trade-off:** a tiny amount of duplicated checking buys defence in depth: even a bug in the graph cannot send the same email twice.

### D8. One `LLMClient.generate(system, user, schema, model)` method
- **Choice:** every provider returns a validated schema instance plus tokens, latency and cost (`LLMResult`). Our instructions go in `system`; email text goes in `user` as data.
- **Alternative:** expose each provider's SDK directly to the agents, or use LangChain chat model wrappers.
- **Trade-off:** we write a thin adapter per provider, but agents stay provider-agnostic, fallback between providers is trivial, and usage metrics are recorded in one place. **Open point for Phase 2:** worker tool loops will need either a `generate_with_tools` method or agents calling tools in plain code; to confirm with Erland before building it.

### D9. FakeLLM uses keyword rules per output schema
- **Choice:** one small handler per schema (triage, summary, draft, schedule) applying keyword rules to the prompt. A tie between intents lowers confidence, which is how the ambiguous email ends up below the 0.6 threshold.
- **Alternative:** canned answers keyed by email id.
- **Trade-off:** keyword rules are cruder than a model, but they react to the input (so edits, preferences and new emails behave sensibly), stay fully deterministic and need no keys. It relies on a simple prompt convention (`From:`, `Subject:`, `Body:`, `[F01] fact`, `Preference key: value`, `Free slot:`, `Due:`) that real prompts will follow anyway.

### D10. Real providers are stubs and their SDKs are imported lazily
- **Choice:** `OpenAIClient` and `AnthropicClient` check for their key at construction and raise `NotImplementedError` on `generate`. The factory imports them only when selected.
- **Alternative:** implement them now.
- **Trade-off:** fake mode installs and runs without any provider SDK. The real implementations are written in a later phase, after checking the installed SDK versions and current docs as the working rules require.

### D11. Default model names
- **Choice:** triage `gpt-5.4-mini` / `claude-haiku-5-5`, drafting `gpt-5.4` / `claude-sonnet-5-5`, all overridable in `.env`.
- **Alternative:** hard-code one model per provider.
- **Trade-off:** small, cheap models for the high-volume classification step and stronger ones only for drafting. The names should be re-checked against the providers' model pages when the real providers are implemented.

### D12. Sample data shape
- **Choice:** 15 emails in 10 threads, all fictional (example.com domains). Every intent is covered, plus a VIP manager with a moved deadline, an ambiguous email (e14), a "bank details changed" email from a lookalike partner domain (e13, `nordwind-billing.example` vs `nordwind.example`), and two partners asking similar questions (e04, e15) so the demo can show a learned drafting style.
- **Alternative:** 6 to 8 threads as first suggested.
- **Trade-off:** spam, newsletters, fraud and the ambiguous email are naturally single-message threads, so 10 threads keeps each case realistic. Timestamps carry explicit offsets (CEST this week, CET after 25 October) so working-hours checks are correct.

### D13. Org facts have stable IDs
- **Choice:** each line of `org_facts.md` starts with an ID like `[F04]`.
- **Alternative:** retrieve facts by text only.
- **Trade-off:** a little formatting discipline, but drafts can cite `facts_used` by ID and the approval view can show exactly which facts a draft relied on (amendment A6).

### D14. PLAN.md section 15 defaults
- **Choice:** until Erland says otherwise, follow the recommended options: LangGraph, Chroma for org facts, JSONL traces with optional Jaeger, and a GitHub repo for submission. The default real provider (OpenAI or Anthropic) is still open; it only matters once the real providers are implemented.
