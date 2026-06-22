"""gpt2 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Gpt2HfTransformersReference(HfTransformersReference):
    """gpt2 local reference for hf_transformers."""

reference = Gpt2HfTransformersReference()
