"""qwen3_5 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Qwen35HfTransformersReference(HfTransformersReference):
    """qwen3_5 local reference for hf_transformers."""

reference = Qwen35HfTransformersReference()
