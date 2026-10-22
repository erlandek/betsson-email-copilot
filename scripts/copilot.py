"""Entry point for the CLI without installing the package: python scripts/copilot.py run"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
