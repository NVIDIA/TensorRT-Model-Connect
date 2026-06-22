"""starcoder2 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Starcoder2HfTransformersReference(HfTransformersReference):
    """starcoder2 local reference for hf_transformers."""

reference = Starcoder2HfTransformersReference()
