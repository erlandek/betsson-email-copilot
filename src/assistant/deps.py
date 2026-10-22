"""Everything the agents and graph nodes need, bundled once.

LangGraph passes this to every node as the run's context (context_schema), so
nodes never import global state and tests can swap in their own pieces.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from assistant.adapters import CalendarAdapter, MailAdapter, MockCalendarAdapter, MockMailAdapter
from assistant.config import Settings
from assistant.llm import LLMClient, get_llm_client
from assistant.llm.fallback import FallbackLLM
from assistant.llm.instrumented import InstrumentedLLM
from assistant.memory.knowledge import KeywordKnowledgeBase, open_knowledge
from assistant.memory.long_term import LongTermMemory
from assistant.observability import Observability


@dataclass
class Deps:
    settings: Settings
    llm: LLMClient
    mail: MailAdapter
    calendar: CalendarAdapter
    memory: LongTermMemory
    knowledge: KeywordKnowledgeBase  # or its Chroma subclass
    obs: Observability


def build_llm(settings: Settings, obs: Observability) -> LLMClient:
    """The configured provider, instrumented, with the other real provider as fallback when its key is set."""
    primary = settings.llm_provider
    llm: LLMClient = InstrumentedLLM(get_llm_client(settings), obs)
    other = {"openai": "anthropic", "anthropic": "openai"}.get(primary)
    if not (settings.llm_fallback and other and getattr(settings, f"{other}_api_key")):
        return llm
    model_map = {settings.model_for(task, primary): settings.model_for(task, other) for task in ("triage", "draft")}
    secondary = InstrumentedLLM(get_llm_client(settings, other), obs)
    return FallbackLLM(llm, secondary, model_map, obs)


def build_deps(settings: Settings, calendar_path: Path | None = None, sent_path: Path | None = None) -> Deps:
    """Wire the mock adapters and memory to the sample data. Writes go to runtime/, never to data/."""
    data, runtime = settings.data_dir, settings.runtime_dir
    calendar_path = calendar_path or runtime / "calendar.json"
    if not calendar_path.exists():
        # Start from the sample calendar; the mock then writes changes to this copy only.
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data / "calendar.json", calendar_path)
    obs = Observability.open(settings.logs_dir, settings.otel_exporter_otlp_endpoint)
    return Deps(
        settings=settings,
        llm=build_llm(settings, obs),
        mail=MockMailAdapter(data / "inbox.json", sent_path or runtime / "sent.json"),
        calendar=MockCalendarAdapter(calendar_path),
        memory=LongTermMemory.open(runtime / "memory.db", data / "seed_memory.json"),
        knowledge=open_knowledge(data / "org_facts.md", settings.knowledge_backend),
        obs=obs,
    )
