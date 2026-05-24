"""Shared project-root resolver for scripts under v14/."""
from __future__ import annotations

import sys
from pathlib import Path

# v14/<subdir>/script.py → repo root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
