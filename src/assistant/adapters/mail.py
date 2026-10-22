"""Mail adapter: an interface plus a mock backed by JSON files.

The agents only ever see MailAdapter, so a real Gmail or Microsoft Graph adapter
could replace the mock without touching agent code. There is deliberately no
delete method: spam and newsletters are labelled and archived, never deleted.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from assistant.schemas import DraftReply, Email, SentRecord, Thread


class MailAdapter(ABC):
    @abstractmethod
    def list_emails(self, include_archived: bool = False) -> list[Email]: ...

    @abstractmethod
    def get_email(self, email_id: str) -> Email: ...

    @abstractmethod
    def get_thread(self, thread_id: str) -> Thread: ...

    @abstractmethod
    def apply_label(self, email_id: str, label: str) -> Email: ...

    @abstractmethod
    def archive(self, email_id: str) -> Email: ...

    @abstractmethod
    def send(self, draft: DraftReply, thread_id: str, action_id: str) -> SentRecord: ...

    @abstractmethod
    def find_sent(self, action_id: str) -> SentRecord | None: ...


class MockMailAdapter(MailAdapter):
    """Reads the inbox from a JSON file and "sends" by appending to sent.json.

    Labels and archiving change only the in-memory copy, so the sample inbox
    file stays untouched between demo runs. Sent mail is written to disk because
    the execute node must be able to see it after a restart (idempotency).
    """

    def __init__(self, inbox_path: Path, sent_path: Path):
        raw = json.loads(Path(inbox_path).read_text(encoding="utf-8"))
        self._emails = {e["id"]: Email.model_validate(e) for e in raw}
        self._sent_path = Path(sent_path)

    def list_emails(self, include_archived: bool = False) -> list[Email]:
        emails = sorted(self._emails.values(), key=lambda e: e.received_at)
        return [e for e in emails if include_archived or not e.archived]

    def get_email(self, email_id: str) -> Email:
        if email_id not in self._emails:
            raise KeyError(f"unknown email id: {email_id}")
        return self._emails[email_id]

    def get_thread(self, thread_id: str) -> Thread:
        messages = [e for e in self.list_emails(include_archived=True) if e.thread_id == thread_id]
        if not messages:
            raise KeyError(f"unknown thread id: {thread_id}")
        return Thread(thread_id=thread_id, subject=messages[0].subject, messages=messages)

    def apply_label(self, email_id: str, label: str) -> Email:
        email = self.get_email(email_id)
        if label not in email.labels:  # labels behave like a set, so repeats are harmless
            email.labels.append(label)
        return email

    def archive(self, email_id: str) -> Email:
        email = self.get_email(email_id)
        email.archived = True
        return email

    def send(self, draft: DraftReply, thread_id: str, action_id: str) -> SentRecord:
        # Second line of defence for idempotency: even if the execute node were
        # called twice, the same action_id is only ever "sent" once.
        existing = self.find_sent(action_id)
        if existing:
            return existing
        record = SentRecord(
            action_id=action_id,
            thread_id=thread_id,
            to=draft.to,
            subject=draft.subject,
            body=draft.body,
            sent_at=datetime.now(timezone.utc),
        )
        records = self._load_sent()
        records.append(record)
        self._save_sent(records)
        return record

    def find_sent(self, action_id: str) -> SentRecord | None:
        return next((r for r in self._load_sent() if r.action_id == action_id), None)

    def _load_sent(self) -> list[SentRecord]:
        if not self._sent_path.exists():
            return []
        raw = json.loads(self._sent_path.read_text(encoding="utf-8"))
        return [SentRecord.model_validate(r) for r in raw]

    def _save_sent(self, records: list[SentRecord]) -> None:
        self._sent_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump(mode="json") for r in records]
        self._sent_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
