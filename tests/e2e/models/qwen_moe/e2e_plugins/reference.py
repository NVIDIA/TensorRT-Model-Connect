"""qwen_moe model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class QwenMoeHfTransformersReference(HfTransformersReference):
    """qwen_moe local reference for hf_transformers."""

reference = QwenMoeHfTransformersReference()
