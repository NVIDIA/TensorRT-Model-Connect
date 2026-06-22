"""xglm model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class XglmHfTransformersReference(HfTransformersReference):
    """xglm local reference for hf_transformers."""

reference = XglmHfTransformersReference()
