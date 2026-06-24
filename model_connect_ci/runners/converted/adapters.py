"""Adapters for normalizing converted-runner outputs."""

from __future__ import annotations

from typing import Any


def normalize_output(value: Any) -> Any:
    """Return a JSON-compatible normalized output when possible."""

    if isinstance(value, dict):
        return {str(key): normalize_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_output(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
