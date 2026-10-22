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

### D8. One `LLMClient` interface with two calls: `generate()` and `next_step()`
- **Choice:** every provider returns a validated schema instance plus tokens, latency and cost (`LLMResult`). `generate()` is one prompt in, one answer out. `next_step()` is one turn of a tool loop: the model either asks for a tool or gives its final answer. Our instructions go in `system`; email text goes in the user message as data.
- **Alternative:** expose each provider's SDK directly to the agents, or use LangChain chat model wrappers.
- **Trade-off:** we write a thin adapter per provider, but agents stay provider-agnostic, fallback between providers is simple, and usage metrics are recorded in one place. The loop itself is our code, so its limits do not depend on the provider (see D15).

### D9. FakeLLM uses keyword rules per output schema
- **Choice:** one small handler per schema (triage, summary, draft, schedule) applying keyword rules to the prompt. A tie between intents lowers confidence, which is how the ambiguous email ends up below the 0.6 threshold. In a tool loop it plays a fixed script of tool calls per schema (D15).
- **Alternative:** canned answers keyed by email id.
- **Trade-off:** keyword rules are cruder than a model, but they react to the input (so edits, preferences and new emails behave sensibly), stay fully deterministic and need no keys. It relies on a simple prompt convention (`From:`, `Subject:`, `Body:`, `[F01] fact`, `Preference key: value`, `Free slot:`, `Due:`) that real prompts will follow anyway.

### D10. Real providers are stubs and their SDKs are imported lazily (superseded by D35)
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
- **Choice:** until Erland says otherwise, follow the recommended options: LangGraph, Chroma for org facts, JSONL traces with optional Jaeger, and a GitHub repo for submission. The default real provider (OpenAI or Anthropic) is still open; it only matters once the real providers are implemented. (Settled by D37: either can be `LLM_PROVIDER`, and the other is the fallback.)

## Phase 2: agents, graph and router

### D15. Hybrid tool use: deterministic pre-fetch plus a bounded tool loop (decided by Erland)
- **Choice:** data that every run needs is fetched in plain code: triage always gets the thread and the sender's contact record, and the summariser always gets the thread. The drafter and scheduler run a real tool-calling loop where the model chooses its tools, with: at most 4 tool calls (`MAX_TOOL_CALLS`), a per-agent allowlist, Pydantic validation of every tool's arguments, and a schema-validated final answer. After the 4th call the model is offered no tools, which forces an answer. Denied or invalid calls are returned to the model as errors and still count towards the limit. FakeLLM emits a fixed script of tool calls so the demo stays deterministic.
- **Alternatives:** (a) tools called only in plain code, with the LLM producing just the final schema; (b) a free tool loop for every agent, including triage.
- **Trade-off:** (a) is cheapest and most predictable but is not really agentic, and it fetches context whether or not the email needs it. (b) spends tokens and latency letting the model rediscover that triage always needs the thread. The hybrid keeps the high-volume step (triage) cheap and fixed, and gives agency where it adds value: deciding which facts and preferences a reply or meeting needs. The cap, allowlist and argument validation bound cost and the damage a prompt-injected model could do. All tools are read-only; sending and creating events are never tools.

### D16. Dependencies reach the nodes through LangGraph's runtime context
- **Choice:** a `Deps` dataclass (settings, LLM client, adapters, memory, knowledge) is passed as `context=` when invoking the graph (`StateGraph(State, context_schema=Deps)`, LangGraph 1.2). Nodes read it via `runtime.context`.
- **Alternative:** module-level globals, or closures created inside `build_graph()`.
- **Trade-off:** this is the documented LangGraph 1.x way. It keeps the graph definition free of setup code, and tests swap any piece (for example a broken LLM) without patching imports. The cost is learning one LangGraph concept.

### D17. Graph state is a `TypedDict` holding Pydantic objects
- **Choice:** `State` lists every field a run can produce (email, triage, summary, draft, schedule, proposal, route, review_reason, errors). Each node returns only the keys it changes.
- **Alternative:** a Pydantic model as the state schema.
- **Trade-off:** a TypedDict is the simplest LangGraph state and the values inside are already validated Pydantic models, so a second layer of validation adds little.

### D18. Routing rules
- **Choice:** the router (a conditional edge) sends results below `CONFIDENCE_THRESHOLD` (0.6) or failed triage to `human_review`; newsletter, spam and fyi to `archive` (label and archive, never delete); reply_needed to summarise, draft, propose; meeting_request and follow_up to schedule, propose. A worker that fails validation twice also routes to `human_review`. The run ends at `propose_action` in this phase; approval and execution are Phase 3.
- **Alternative:** route fyi to the inbox untouched instead of archiving.
- **Trade-off:** archiving fyi follows PLAN 3.2 and keeps the inbox for things that need action. It is reversible, and the email keeps its `fyi` label.

### D19. Labels are topic tags, applied automatically
- **Choice:** labels describe what an email is about (Finance, Legal, Partners, Internal, Reporting, IT, Meetings), never the intent, which is its own field. The triage prompt asks for one to three such tags, and `apply_priority_rules()` drops any tag that repeats an intent name. The triage node applies tags and rule warnings straight away (low risk, reversible), even when the intent goes to human review. Archived spam and newsletters also get a category label ("Spam", "Newsletters") so they stay findable.
- **Alternative:** use the intent as the label (the first version did this).
- **Trade-off:** the intent label duplicated information and was not useful in a mailbox. Tags are what a person files by. On an uncertain email a topic tag is still usually right even when the intent is not; "Needs review" marks the uncertainty.

### D20. A known contact classified as spam goes to human review
- **Choice:** amendment A3 says a known contact or VIP is never spam. Instead of guessing the real intent, the rule changes the intent to fyi, caps confidence at 0.5 and adds "Spam override: known sender", so a human looks at it.
- **Alternative:** silently relabel it as reply_needed.
- **Trade-off:** one extra item in the review queue, against the risk of auto-drafting a reply to a compromised colleague account.

### D21. Contacts, preferences and org facts are read-only stand-ins until Phase 3 (superseded by D30 and D31)
- **Choice:** `LongTermMemory` loads contacts, preferences and known newsletter senders from `data/seed_memory.json`; `KnowledgeBase` searches `org_facts.md` by keyword overlap (at least two shared words). Phase 3 moves memory to SQLite and knowledge to Chroma behind the same methods.
- **Alternative:** build SQLite and Chroma now.
- **Trade-off:** keeps this phase to agents and routing, as the plan's time plan intends. Keyword search is crude (it misses synonyms such as "docs" vs "documentation"), which is exactly the point the Chroma swap will demonstrate.

### D22. Scheduling dates are computed in code, in the user's time zone
- **Choice:** the scheduler gets a search window (next working week, Monday to Friday, `TIMEZONE=Europe/Stockholm`) computed in code; `check_availability` returns merged free periods within working hours from preferences.
- **Alternative:** let the model work out "next week" itself.
- **Trade-off:** models are unreliable at date arithmetic and time zones (this week is CEST, next week CET). Computing the window removes a whole class of errors and keeps the tool output to a few lines.
- **Later:** the dry run in Phase 6 found that a follow-up reminder for e09 was due at 09:00+02:00, which is 08:00 in Stockholm after the clocks change on 25 October. The default due time (09:00 local, a week after the email) is now computed in code the same way and given to the model.

## Fixes before Phase 3

### D23. Lookalike-domain check in the fraud rules
- **Choice:** for a sender who is not in the contact book, compare their domain with the domains of known contacts. The part before the top-level domain is compared: flagged if one contains the other ("nordwind-billing" contains "nordwind") or they are at least 80% similar by `difflib.SequenceMatcher` ("examp1e" vs "example"). A match caps confidence at 0.5 and adds "Lookalike domain: verify sender", even without payment words.
- **Alternative:** edit distance (Levenshtein) only, or a confusables/homoglyph library.
- **Trade-off:** standard library, a few lines, easy to explain, and it catches the two common tricks (an added word, a swapped character). It can flag a legitimate subdomain such as `mail.nordwind.example`; that only costs a human review. Homoglyphs in Unicode domains (Cyrillic "а") would need a confusables table: future work.

### D24. Every datetime is time-zone aware, and only unread mail is processed
- **Choice:** all schema datetimes use Pydantic's `AwareDatetime`, which rejects naive values. Sample data carries Stockholm offsets (+02:00 before 25 October, +01:00 after). Scheduling converts to `TIMEZONE` (Europe/Stockholm) before comparing with working hours. `Email.unread` was added, and `emails_to_process` takes the latest message of each thread only if it is unread and was not sent by the user.
- **Alternative:** naive datetimes in local time; processing every inbound email.
- **Trade-off:** aware datetimes cost an offset in the data, but the week of the demo crosses the switch from summer time, which is exactly where naive times go wrong. Processing one message per thread avoids triaging the same conversation three times; earlier messages still reach the agents as thread context.

### D25. Working hours are enforced in code, twice
- **Choice:** `check_availability` only returns free time inside the working hours from preferences (minutes included, e.g. 10:30-15:00). After the scheduler answers, `check_proposal()` verifies the event is on a weekday, inside working hours and free in the calendar; if not, the email goes to human review.
- **Alternative:** trust the model to pick from the free times it was shown.
- **Trade-off:** a real model can still invent a time; checking costs a few lines and makes "the scheduler respects working hours" a guarantee, not a hope.

## Phase 3: approval, execution, memory and learning

### D26. Approval is an `interrupt()` inside a `human_approval` node
- **Choice:** after `propose_action`, the graph routes to `human_approval` when `requires_approval()` says so (always, with `APPROVAL_MODE=all`). The node builds the approval view (triage, summary, proposal, facts used: amendment A6) and calls `interrupt(view)`. The run stops and the state is checkpointed. Resuming with `Command(resume=decision)` re-runs the node and `interrupt()` returns the decision. An invalid decision (for example an edit that turns an email into a task) is not acted on; the node asks again with the error in the view.
- **Alternative:** `interrupt_before=["execute"]` on compile, with the decision written into the state from outside.
- **Trade-off:** `interrupt()` keeps the question and the validation of the answer in one readable node and is the current LangGraph recommendation. The catch, worth knowing in an interview: the node re-runs from the top on resume, so nothing before `interrupt()` may have side effects.

### D27. Deterministic `action_id` for exactly-once execution (amendment A7)
- **Choice:** before acting, `execute` computes `action_id = sha256(email id, action type, final payload)` and asks the adapter whether that id already exists (`find_sent`, `find_by_action_id`). Only if not does it send or create, storing the id with the record. The adapters also refuse duplicates (D7).
- **Alternative:** a random UUID created in the node and saved to the state before acting.
- **Trade-off:** a random id saved in the same node is lost if the process dies between acting and checkpointing, and the retry would make a new id and act twice; a deterministic id survives that crash. An edited payload is a genuinely different action and gets a different id. Re-approving an identical action for the same email is treated as a duplicate, which is the safe default.

### D28. One checkpointer thread per email, in SQLite
- **Choice:** `SqliteSaver` in `runtime/checkpoints.db`, with `thread_id` = email id. `process_email` looks at the saved state first: finished emails are not reprocessed, emails waiting for approval are resumed (even in a new process), new emails start fresh.
- **Alternative:** `thread_id` = email thread id, as first written in PLAN section 5.
- **Trade-off:** a mail thread receives new messages over time, and reusing its run would carry stale keys (an old draft, an old decision) into the new run. One run per email is clean; the thread's history still reaches the agents through `get_thread`.

### D29. Checkpoints only deserialise our own schema classes
- **Choice:** the checkpointer's serializer is given an allowlist of the Pydantic models and enums in `assistant.schemas` (`allowed_msgpack_modules`).
- **Alternative:** the default, which accepts any class and warns.
- **Trade-off:** LangGraph 1.x warns that loading arbitrary classes from a checkpoint database is a code-execution risk and will block it in a future version. The allowlist removes the warning and the risk, and needs no maintenance because it is built from the schemas module.

### D30. Long-term memory in SQLite, written only from human decisions (amendment A8)
- **Choice:** `runtime/memory.db` with four tables: preferences (key, value, source seeded/learned, updated_at), contacts, feedback (decision, our draft before, the human's version after, comment) and newsletter senders. Seeded from `data/seed_memory.json` the first time. Writes happen only in `update_memory`, after a human decision: feedback for every decision, learned preferences from edits, and the recipient of an approved reply becomes a contact ("replied to").
- **Alternative:** PLAN section 5 said new senders are added to contacts automatically.
- **Trade-off:** adding every sender would let any email make itself a "known contact", which would switch off the fraud and lookalike checks (they only apply to unknown senders). Adding a contact only when the user chooses to reply follows A8. A test checks that processing emails without decisions writes nothing.

### D31. Chroma with a local embedding model for org facts
- **Choice:** `ChromaKnowledgeBase` builds an in-memory Chroma collection from `org_facts.md` at start-up, using Chroma's built-in all-MiniLM-L6-v2 model (runs locally, downloaded once, about 80 MB, no API key). Facts further than cosine distance 0.65 are dropped (tuned on the sample inbox). `KNOWLEDGE_BACKEND=keyword` switches to word overlap; most tests use it to stay fast.
- **Alternative:** an embeddings API (needs a key, breaks the no-keys demo), or a persistent Chroma database.
- **Trade-off:** semantic search finds "docs" in a fact about "documentation", which word overlap misses. The cost is a one-time download and a heavier dependency. Rebuilding from the markdown file each start keeps the store and the file in sync and costs well under a second for ten facts.

### D32. Learning loop with a fixed set of learnable keys
- **Choice:** when the human edits a draft, `learn_from_edit()` asks the small model for `LearnedPreferences`: updates limited to sign_off, reply_length, greeting and tone (max four, 100 characters each). FakeLLM uses two diff rules: a new last line is a sign-off; a draft cut by over 30% means short replies. The drafter's `get_preferences` tool returns the preferences plus the two most recent edits as examples.
- **Alternative:** store free-text "lessons" from the model, or only store examples.
- **Trade-off:** fixed keys can be shown, explained and reset, and a manipulated edit cannot create new kinds of memory. Examples carry nuance the keys miss, so the drafter gets both.

### D33. A small Typer CLI, launched through `scripts/copilot.py`
- **Choice:** `run` processes unread email and asks approve, edit, reject or skip for each proposal (skip leaves it pending for the next run); `memory` and `outbox` show what was learned, sent and scheduled. Editing a draft is multi-line input ending with a line containing ".". `process_email(..., decide=...)` takes any function, so tests and the notebook pass scripted decisions.
- **Alternative:** a web UI, or opening the user's text editor for edits.
- **Trade-off:** a terminal is enough to demonstrate the loop and works the same on Windows and Linux. Typer in this version no longer exposes an editor helper, and multi-line input avoids depending on one.

### D34. The human review queue is a mailbox label
- **Choice:** low-confidence, fraud, lookalike and failed runs end in `human_review`, which labels the email "Needs review" and records the reason in the run's state.
- **Alternative:** a separate review table or an interrupt that blocks the run.
- **Trade-off:** the user already lives in their mailbox, and nothing outgoing is proposed for these emails, so there is nothing to approve yet. A dedicated review UI is future work.

### D35. Real providers: native structured output and native tool calling, our loop in charge
- **Choice:** `OpenAIClient` (Responses API) and `AnthropicClient` (Messages API) implement the same two calls as FakeLLM. The answer schema is sent as the provider's strict JSON-schema output format and each tool as a strict native tool built from its `args_model`; we still validate both with Pydantic, so the retry and human-review path is the same for every provider. One tool call per turn (parallel calls off). Past the limit the tools are still described but `tool_choice` is "none", so the model must answer; forcing a call is not used because current Anthropic models reject it. The provider's own turn is replayed unchanged, which keeps Anthropic thinking blocks and OpenAI encrypted reasoning valid. OpenAI runs with `store=False` so email text is not kept server-side. Anthropic runs at effort "low". Connection, rate-limit and server errors become `LLMError` (one retry, then human review); a bad key or bad request stops the run, since it would fail for every email. Cost comes from a small price table checked on 2026-10-22. Tests use stand-in SDK clients; no keys are needed.
- **Alternative:** LangChain chat-model wrappers with `with_structured_output`, or each SDK's own tool runner.
- **Trade-off:** wrappers and runners would hide the loop, but our loop is where the call limit, allowlists and argument validation live (D15), so it has to stay ours. About 120 lines per provider is small enough to explain line by line. Neither provider has been run against the live API from this environment (no keys here); the request shapes were checked by sending them through the real SDKs with a mocked HTTP transport.

### D36. Observability lives on Deps: structlog JSON logs, OpenTelemetry spans, a small metrics registry
- **Choice:** `Observability` holds a structlog logger (`logs/app.jsonl`, errors copied to `logs/errors.jsonl`), an OpenTelemetry tracer (`logs/traces.jsonl`, plus OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set) and an in-process metrics registry (`logs/metrics.json`). It is created in `build_deps` and reaches every node through the run context, like the adapters. One trace per `process_email` call, a span per node (a wrapper added in `build_graph`), per LLM call (`InstrumentedLLM`, named and tagged with the OpenTelemetry GenAI conventions: `chat <model>`, `gen_ai.usage.input_tokens`, ...) and per tool call (`execute_tool <name>`). Log lines carry the run, email, trace and span ids. Spans are written as each one ends, so a run stopped at an approval prompt is still on disk. A paused `interrupt()` is marked "waiting for human", not as an error; errors a node handles (an `AgentError` that routes to review) are logged, counted and mark the span as failed. `scripts/show_trace.py` prints a waterfall.
- **Alternative:** register the tracer and logger globally (the usual `trace.set_tracer_provider` and `structlog.configure`), use OpenTelemetry metrics, or a hosted tool such as Langfuse or LangSmith.
- **Trade-off:** keeping them on Deps means tests get their own files and nothing leaks between runs, at the cost of passing `deps` where spans are made (it already was). The metrics registry is about 60 lines and needs no collector; its names map one to one onto Prometheus or OpenTelemetry metrics later. Prompts and email bodies are deliberately not recorded: they are personal data, and tokens, latency and outcome are what is needed to operate the system. A test checks that no email text reaches the trace file.

### D37. Retries in the SDKs, then fallback to the other provider
- **Choice:** both SDKs already retry connection errors, timeouts, 429 and 5xx with exponential backoff; we set `LLM_MAX_RETRIES` (2) and `LLM_TIMEOUT_SECONDS` (60) on them instead of writing our own loop. What still fails becomes `LLMError`, and `FallbackLLM` sends the same call to the other real provider with that provider's model for the same task (gpt-5.4-mini to claude-haiku-5-5, gpt-5.4 to claude-sonnet-5-5, and back). It is on when `LLM_FALLBACK=true` and the other key is set. Invalid output does not switch provider: the agent retries once with the error and then sends the email to human review. Each provider attempt gets its own span, so a fallback shows in the waterfall as a failed call followed by a good one, and `llm_fallbacks_total` counts it. The step limit per run is LangGraph's `recursion_limit` (25) plus the tool loop's call limit.
- **Alternative:** our own retry decorator (for example with tenacity), or a circuit breaker that stops trying the primary for a while.
- **Trade-off:** the SDK retry logic already honours `retry-after` headers and is tested by the vendors, so a second layer would only multiply waiting time. Without a circuit breaker, a provider that is fully down costs one failed attempt per call before the fallback; for a single user's inbox that is acceptable and much simpler to explain. Mid tool loop the fallback provider sees the earlier tool calls rebuilt in its own format, without the first provider's hidden reasoning, which is fine because each call stands on the transcript.

## Phase 5: demo, evaluation and documentation

### D38. The evaluation measures the whole triage step, and counts the mistakes nobody would see
- **Choice:** `scripts/eval_classifier.py` runs `triage_email()` (pre-filter, model, rules) on 20 labelled emails kept apart from the demo inbox, each a one-message thread. Besides accuracy, per-intent precision and recall, a confusion matrix and accuracy by confidence bucket, it reports "wrong, not reviewed": wrong intents with confidence above the threshold and no fraud warning, which the router would act on without a human. Models are given as `provider:model`, fallback is off, and several models print in one table. The labels carry a one-line reason, and some cases are deliberately hard (a sales pitch asking for a call, phishing from a lookalike brand).
- **Alternative:** evaluate only the raw model output, reuse the demo inbox, or report accuracy alone.
- **Trade-off:** measuring the full step is what the user experiences, but it mixes model and rule effects; the per-intent table and the misclassified list show where an error came from. A separate set avoids tuning the fake model's keywords to the test. Twenty emails are enough to compare models and catch regressions, not to claim a general accuracy. The fake model scores 55%: it is a keyword baseline, and its numbers are kept in the README as they are.

### D39. The demo notebook drives the same code as the CLI, with scripted decisions
- **Choice:** `notebooks/demo.ipynb` calls `process_email()` with a `decide` function that edits e04, rejects e03 and approves the rest, so the run is reproducible. It writes to a throwaway runtime folder and to `logs/demo/`; `sample_logs/` is a copy of one run, plus two trace waterfalls and the evaluation report. A last section simulates a provider outage with a broken fake provider: without a fallback the email goes to review and the error lands in `errors.jsonl`, with one the run continues.
- **Alternative:** a separate demo script with its own logic, or a recorded screencast.
- **Trade-off:** nothing in the notebook is special-cased, so what it shows is what the code does. The outage is simulated, because a real outage cannot be produced on demand; the fallback path between real providers is covered by unit tests.
