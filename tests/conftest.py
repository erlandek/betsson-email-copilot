import shutil
from pathlib import Path

import pytest

from assistant.adapters import MockCalendarAdapter, MockMailAdapter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def mail(tmp_path: Path) -> MockMailAdapter:
    return MockMailAdapter(DATA_DIR / "inbox.json", tmp_path / "sent.json")


@pytest.fixture
def calendar(tmp_path: Path) -> MockCalendarAdapter:
    # Work on a copy so tests never modify the sample calendar.
    path = tmp_path / "calendar.json"
    shutil.copy(DATA_DIR / "calendar.json", path)
    return MockCalendarAdapter(path)
