"""Long-term memory: contacts, preferences and known newsletter senders.

Phase 2 keeps this read-only and in memory, loaded from data/seed_memory.json,
so the agents can already look up contacts and preferences. Phase 3 moves the
same methods onto SQLite and adds writes from human decisions (never directly
from email content, to prevent memory poisoning).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from assistant.schemas import ContactRecord, PreferenceRecord


class LongTermMemory:
    def __init__(self, contacts: list[ContactRecord], preferences: list[PreferenceRecord], newsletter_senders: set[str]):
        self._contacts = {c.email.lower(): c for c in contacts}
        self._preferences = {p.key: p for p in preferences}
        self._newsletter_senders = {s.lower() for s in newsletter_senders}

    @classmethod
    def from_seed(cls, path: Path) -> "LongTermMemory":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        return cls(
            contacts=[ContactRecord.model_validate(c) for c in raw.get("contacts", [])],
            preferences=[PreferenceRecord(source="seeded", updated_at=now, **p) for p in raw.get("preferences", [])],
            newsletter_senders=set(raw.get("newsletter_senders", [])),
        )

    def get_contact(self, email: str) -> ContactRecord | None:
        return self._contacts.get(email.lower())

    def get_preferences(self) -> dict[str, str]:
        return {key: p.value for key, p in self._preferences.items()}

    def is_known_newsletter(self, sender: str) -> bool:
        return sender.lower() in self._newsletter_senders
