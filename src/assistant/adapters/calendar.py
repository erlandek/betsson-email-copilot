"""Calendar adapter: an interface plus a mock backed by one JSON file.

calendar.json holds {"events": [...], "tasks": [...]}. Unlike the mailbox, the
mock writes every change to disk, so the demo can show the new event in the file.
"""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

from assistant.schemas import CalendarEvent, CalendarTask, ScheduleProposal


class CalendarAdapter(ABC):
    @abstractmethod
    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]: ...

    @abstractmethod
    def is_free(self, start: datetime, end: datetime) -> bool: ...

    @abstractmethod
    def free_slots(
        self, start: datetime, end: datetime, duration: timedelta, day_start: int = 9, day_end: int = 17
    ) -> list[tuple[datetime, datetime]]: ...

    @abstractmethod
    def create_event(self, proposal: ScheduleProposal, action_id: str) -> CalendarEvent: ...

    @abstractmethod
    def create_task(self, proposal: ScheduleProposal, action_id: str) -> CalendarTask: ...

    @abstractmethod
    def find_by_action_id(self, action_id: str) -> CalendarEvent | CalendarTask | None: ...


class MockCalendarAdapter(CalendarAdapter):
    def __init__(self, path: Path):
        self._path = Path(path)

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        events, _ = self._load()
        overlapping = [e for e in events if e.start < end and e.end > start]
        return sorted(overlapping, key=lambda e: e.start)

    def is_free(self, start: datetime, end: datetime) -> bool:
        return not self.list_events(start, end)

    def free_slots(
        self, start: datetime, end: datetime, duration: timedelta, day_start: int = 9, day_end: int = 17
    ) -> list[tuple[datetime, datetime]]:
        """Free slots of `duration` on weekdays within working hours, in 30-minute steps.

        A simple scan is enough for a single user's week and is easy to test.
        """
        slots = []
        cursor = start
        step = timedelta(minutes=30)
        while cursor + duration <= end:
            slot_end = cursor + duration
            in_hours = cursor.hour >= day_start and (slot_end.hour, slot_end.minute) <= (day_end, 0)
            same_day = slot_end.date() == cursor.date()
            if cursor.weekday() < 5 and in_hours and same_day and self.is_free(cursor, slot_end):
                slots.append((cursor, slot_end))
            cursor += step
        return slots

    def create_event(self, proposal: ScheduleProposal, action_id: str) -> CalendarEvent:
        if proposal.kind != "event":
            raise ValueError("create_event needs a proposal of kind 'event'")
        existing = self.find_by_action_id(action_id)
        if isinstance(existing, CalendarEvent):
            return existing  # already created on an earlier attempt
        event = CalendarEvent(
            id=f"evt-{uuid.uuid4().hex[:8]}",
            title=proposal.title,
            start=proposal.start,
            end=proposal.end,
            attendees=proposal.attendees,
            action_id=action_id,
        )
        events, tasks = self._load()
        events.append(event)
        self._save(events, tasks)
        return event

    def create_task(self, proposal: ScheduleProposal, action_id: str) -> CalendarTask:
        if proposal.kind != "task":
            raise ValueError("create_task needs a proposal of kind 'task'")
        existing = self.find_by_action_id(action_id)
        if isinstance(existing, CalendarTask):
            return existing
        task = CalendarTask(id=f"task-{uuid.uuid4().hex[:8]}", title=proposal.title, due=proposal.due, action_id=action_id)
        events, tasks = self._load()
        tasks.append(task)
        self._save(events, tasks)
        return task

    def find_by_action_id(self, action_id: str) -> CalendarEvent | CalendarTask | None:
        events, tasks = self._load()
        return next((item for item in [*events, *tasks] if item.action_id == action_id), None)

    def _load(self) -> tuple[list[CalendarEvent], list[CalendarTask]]:
        if not self._path.exists():
            return [], []
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        events = [CalendarEvent.model_validate(e) for e in raw.get("events", [])]
        tasks = [CalendarTask.model_validate(t) for t in raw.get("tasks", [])]
        return events, tasks

    def _save(self, events: list[CalendarEvent], tasks: list[CalendarTask]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": [e.model_dump(mode="json") for e in events],
            "tasks": [t.model_dump(mode="json") for t in tasks],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
