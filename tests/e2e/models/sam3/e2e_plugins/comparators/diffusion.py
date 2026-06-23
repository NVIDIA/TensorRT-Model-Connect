"""Shared diffusion comparator placeholder.

Diffusion media comparison policy is model-owned under
``tests/e2e/models/<family>/e2e_plugins/comparators/diffusion.py``. Generic
metric helpers may remain in sibling helper modules, but this module must not
register concrete comparison behavior.
"""

from __future__ import annotations

plugin = None
