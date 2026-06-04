"""Shared fixtures for diff framework self-tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_tools_to_path():
    """Ensure tools/ and the Python package are importable."""
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root / "tools", repo_root / "python"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
