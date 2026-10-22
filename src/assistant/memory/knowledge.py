"""Organisational facts the drafter can search and cite by ID.

Phase 2 uses plain keyword overlap: with ten short facts it is predictable and
needs no embedding model. Phase 3 swaps in Chroma behind the same search() method.
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


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


class KnowledgeBase:
    def __init__(self, facts: list[Fact]):
        self._facts = facts

    @classmethod
    def from_markdown(cls, path: Path) -> "KnowledgeBase":
        # Each fact is a line like "- [F04] Partner support hours are ..."
        text = Path(path).read_text(encoding="utf-8")
        facts = [Fact(id=m[0], text=m[1].strip()) for m in re.findall(r"^- \[(F\d+)\]\s*(.+)$", text, re.MULTILINE)]
        return cls(facts)

    def search(self, query: str, k: int = 3, min_overlap: int = 2) -> list[Fact]:
        # Requiring two shared words filters out facts that share one common word by chance.
        query_words = _words(query)
        scored = [(len(query_words & _words(f.text)), f) for f in self._facts]
        scored = [(score, f) for score, f in scored if score >= min_overlap]
        scored.sort(key=lambda sf: sf[0], reverse=True)  # stable: ties keep file order
        return [f for _, f in scored[:k]]
