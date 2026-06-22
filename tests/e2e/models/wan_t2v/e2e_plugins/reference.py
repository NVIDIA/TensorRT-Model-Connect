"""wan_t2v model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_diffusers import HfDiffusersReference


class WanT2vHfDiffusersReference(HfDiffusersReference):
    """wan_t2v local reference for hf_diffusers."""

reference = WanT2vHfDiffusersReference()
