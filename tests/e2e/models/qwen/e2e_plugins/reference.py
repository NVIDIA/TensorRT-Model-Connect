"""qwen model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference
from .references.invariant_only import InvariantOnlyReference


class QwenHfTransformersReference(HfTransformersReference):
    """qwen local reference for hf_transformers."""


class QwenInvariantOnlyReference(InvariantOnlyReference):
    """qwen local reference for invariant_only."""

reference = [
    QwenHfTransformersReference(),
    QwenInvariantOnlyReference(),
]
