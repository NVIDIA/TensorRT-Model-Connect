"""flux model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_diffusers import HfDiffusersReference


class FluxHfDiffusersReference(HfDiffusersReference):
    """flux local reference for hf_diffusers."""

reference = FluxHfDiffusersReference()
