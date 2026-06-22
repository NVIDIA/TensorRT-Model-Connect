"""gpt_neox model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class GptNeoxHfTransformersReference(HfTransformersReference):
    """gpt_neox local reference for hf_transformers."""

reference = GptNeoxHfTransformersReference()
