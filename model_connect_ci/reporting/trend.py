"""Trend storage hook placeholders for nightly and weekly mutation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_trend_snapshot(result_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a local trend snapshot that future CI jobs can archive."""

    path = result_dir / "trend_snapshot.json"
    from model_connect_ci.reporting.json_reports import write_json

    write_json(path, payload)
    return path
