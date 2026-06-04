"""Cosmos-Predict2 family package.

Exposes a module-level ``plugin`` attribute consumed by the family
auto-discovery loader in ``families/__init__.py``.
"""

from __future__ import annotations

from .plugin import plugin

__all__ = ["plugin"]
