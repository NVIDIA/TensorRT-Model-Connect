"""qwen3_omni model-owned E2E reference plugins."""

from __future__ import annotations

from .references.invariant_only import InvariantOnlyReference


class Qwen3OmniInvariantOnlyReference(InvariantOnlyReference):
    """qwen3_omni local reference for invariant_only."""


reference = Qwen3OmniInvariantOnlyReference()
