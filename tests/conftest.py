import shutil
from pathlib import Path

import pytest

from assistant.adapters import MockCalendarAdapter, MockMailAdapter
from assistant.config import Settings
from assistant.deps import Deps, build_deps

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore settings left in the shell, such as $env:LLM_PROVIDER, so tests always run on fake."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def mail(tmp_path: Path) -> MockMailAdapter:
    return MockMailAdapter(DATA_DIR / "inbox.json", tmp_path / "sent.json")


@pytest.fixture
def calendar(tmp_path: Path) -> MockCalendarAdapter:
    # Work on a copy so tests never modify the sample calendar.
    path = tmp_path / "calendar.json"
    shutil.copy(DATA_DIR / "calendar.json", path)
    return MockCalendarAdapter(path)


@pytest.fixture
def deps(tmp_path: Path) -> Deps:
    """Fake LLM, mock adapters on the sample data, all writes under tmp_path."""
    # Keyword knowledge search keeps the suite fast; test_memory.py covers Chroma.
    settings = Settings(_env_file=None, data_dir=DATA_DIR, runtime_dir=tmp_path / "runtime",
                        logs_dir=tmp_path / "logs", knowledge_backend="keyword")
    deps = build_deps(settings)
    yield deps
    deps.obs.close()
