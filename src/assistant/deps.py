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
from assistant.memory.knowledge import KnowledgeBase
from assistant.memory.long_term import LongTermMemory


@dataclass
class Deps:
    settings: Settings
    llm: LLMClient
    mail: MailAdapter
    calendar: CalendarAdapter
    memory: LongTermMemory
    knowledge: KnowledgeBase


def build_deps(settings: Settings, calendar_path: Path | None = None, sent_path: Path | None = None) -> Deps:
    """Wire the mock adapters to the sample data. Writes go to runtime/, never to data/."""
    data, runtime = settings.data_dir, settings.runtime_dir
    calendar_path = calendar_path or runtime / "calendar.json"
    if not calendar_path.exists():
        # Start from the sample calendar; the mock then writes changes to this copy only.
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data / "calendar.json", calendar_path)
    return Deps(
        settings=settings,
        llm=get_llm_client(settings),
        mail=MockMailAdapter(data / "inbox.json", sent_path or runtime / "sent.json"),
        calendar=MockCalendarAdapter(calendar_path),
        memory=LongTermMemory.from_seed(data / "seed_memory.json"),
        knowledge=KnowledgeBase.from_markdown(data / "org_facts.md"),
    )
