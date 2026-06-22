"""dpr model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class DprHfTransformersReference(HfTransformersReference):
    """dpr local reference for hf_transformers."""

reference = DprHfTransformersReference()
