"""bark model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class BarkHfTransformersReference(HfTransformersReference):
    """bark local reference for hf_transformers."""

reference = BarkHfTransformersReference()
