"""convbert model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class ConvbertHfTransformersReference(HfTransformersReference):
    """convbert local reference for hf_transformers."""

reference = ConvbertHfTransformersReference()
