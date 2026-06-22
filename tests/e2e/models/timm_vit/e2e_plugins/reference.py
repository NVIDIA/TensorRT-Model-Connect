"""timm_vit model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class TimmVitHfTransformersReference(HfTransformersReference):
    """timm_vit local reference for hf_transformers."""

reference = TimmVitHfTransformersReference()
