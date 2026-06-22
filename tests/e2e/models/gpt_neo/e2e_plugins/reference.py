"""gpt_neo model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class GptNeoHfTransformersReference(HfTransformersReference):
    """gpt_neo local reference for hf_transformers."""

reference = GptNeoHfTransformersReference()
