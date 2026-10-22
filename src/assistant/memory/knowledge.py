"""Organisational facts the drafter can search and cite by ID.

Two interchangeable stores with the same search() method:
- ChromaKnowledgeBase (default): semantic search with Chroma's built-in local
  embedding model (all-MiniLM-L6-v2, downloaded once, no API key). It finds
  "docs" when the fact says "documentation".
- KeywordKnowledgeBase: plain word overlap. No model download, used by most
  tests to keep them fast, and a fallback with KNOWLEDGE_BACKEND=keyword.
"""

import re
from pathlib import Path

from pydantic import BaseModel

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "for", "from", "hi", "how", "i", "if",
    "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "so", "that", "the", "this", "to", "us",
    "we", "what", "when", "with", "you", "your", "thanks", "hello",
}


class Fact(BaseModel):
    id: str
    text: str


def load_facts(path: Path) -> list[Fact]:
    # Each fact is a line like "- [F04] Partner support hours are ..."
    text = Path(path).read_text(encoding="utf-8")
    return [Fact(id=m[0], text=m[1].strip()) for m in re.findall(r"^- \[(F\d+)\]\s*(.+)$", text, re.MULTILINE)]


class KeywordKnowledgeBase:
    def __init__(self, facts: list[Fact]):
        self._facts = facts

    @classmethod
    def from_markdown(cls, path: Path) -> "KeywordKnowledgeBase":
        return cls(load_facts(path))

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self._facts if f.id == fact_id), None)

    def search(self, query: str, k: int = 3, min_overlap: int = 2) -> list[Fact]:
        # Requiring two shared words filters out facts that share one common word by chance.
        query_words = _words(query)
        scored = [(len(query_words & _words(f.text)), f) for f in self._facts]
        scored = [(score, f) for score, f in scored if score >= min_overlap]
        scored.sort(key=lambda sf: sf[0], reverse=True)  # stable: ties keep file order
        return [f for _, f in scored[:k]]


class ChromaKnowledgeBase(KeywordKnowledgeBase):
    # Cosine distance above this means "not about the same thing". Tuned on the
    # sample inbox: relevant facts score 0.42 to 0.64, unrelated ones 0.67 and up.
    MAX_DISTANCE = 0.65

    def __init__(self, facts: list[Fact]):
        import chromadb  # imported here so the keyword store works without it

        super().__init__(facts)
        # Rebuilt in memory from org_facts.md at start-up: ten facts take a
        # moment, and the store can never drift from the file.
        client = chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection("org_facts", configuration={"hnsw": {"space": "cosine"}})
        self._collection.upsert(ids=[f.id for f in facts], documents=[f.text for f in facts])

    def search(self, query: str, k: int = 3, min_overlap: int = 2) -> list[Fact]:
        result = self._collection.query(query_texts=[query], n_results=min(k, len(self._facts)))
        hits = zip(result["ids"][0], result["documents"][0], result["distances"][0])
        return [Fact(id=i, text=t) for i, t, distance in hits if distance <= self.MAX_DISTANCE]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def open_knowledge(path: Path, backend: str) -> KeywordKnowledgeBase:
    facts = load_facts(path)
    return ChromaKnowledgeBase(facts) if backend == "chroma" else KeywordKnowledgeBase(facts)
