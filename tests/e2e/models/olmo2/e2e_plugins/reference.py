"""olmo2 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Olmo2HfTransformersReference(HfTransformersReference):
    """olmo2 local reference for hf_transformers."""

reference = Olmo2HfTransformersReference()
