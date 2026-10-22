from datetime import datetime, timezone

import pytest
from conftest import DATA_DIR
from pydantic import ValidationError

from assistant.memory.knowledge import ChromaKnowledgeBase, KeywordKnowledgeBase, load_facts
from assistant.memory.long_term import LongTermMemory
from assistant.schemas import ContactRecord, FeedbackRecord

SEED = DATA_DIR / "seed_memory.json"


def test_seeded_once_and_persisted(tmp_path):
    db = tmp_path / "memory.db"
    memory = LongTermMemory.open(db, SEED)
    memory.set_preference("sign_off", "Cheers, Erland", source="learned")
    reopened = LongTermMemory.open(db, SEED)  # must not re-seed over the learned value
    assert reopened.get_preferences()["sign_off"] == "Cheers, Erland"
    assert reopened.get_contact("SOFIA.LIND@example.com").vip  # case-insensitive lookup
    assert reopened.is_known_newsletter("news@cloudtools.example")


def test_preference_source_is_validated():
    memory = LongTermMemory(":memory:")
    with pytest.raises(ValidationError):
        memory.set_preference("sign_off", "x", source="email")


def test_feedback_and_recent_edits():
    memory = LongTermMemory(":memory:")
    now = datetime.now(timezone.utc)
    for i, decision in enumerate(["edit", "approve", "edit", "edit"]):
        memory.add_feedback(FeedbackRecord(email_id=f"e{i}", decision=decision, draft_before="a",
                                           draft_after="b" if decision == "edit" else None, created_at=now))
    assert [f.email_id for f in memory.recent_edits(2)] == ["e3", "e2"]
    assert len(memory.all_feedback()) == 4


def test_known_domains_include_new_contacts():
    memory = LongTermMemory(":memory:")
    memory.upsert_contact(ContactRecord(email="anna@fjord.example"))
    assert memory.known_domains() == {"fjord.example"}


def test_keyword_search_needs_two_shared_words():
    kb = KeywordKnowledgeBase(load_facts(DATA_DIR / "org_facts.md"))
    assert [f.id for f in kb.search("sandbox credentials please")][:1] == ["F02"]
    assert kb.search("hello weather") == []
    assert kb.get("F05").text.startswith("Bank detail changes")


def test_chroma_search_understands_meaning():
    pytest.importorskip("chromadb")
    kb = ChromaKnowledgeBase(load_facts(DATA_DIR / "org_facts.md"))
    ids = [f.id for f in kb.search("Where can I find the API docs?")]
    assert "F03" in ids  # the fact says "documentation", not "docs"
    assert kb.search("recipe for cinnamon buns") == []
