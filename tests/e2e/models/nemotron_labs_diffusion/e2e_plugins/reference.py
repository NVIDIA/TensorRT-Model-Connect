"""nemotron_labs_diffusion model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class NemotronLabsDiffusionHfTransformersReference(HfTransformersReference):
    """nemotron_labs_diffusion local reference for hf_transformers."""

reference = NemotronLabsDiffusionHfTransformersReference()
