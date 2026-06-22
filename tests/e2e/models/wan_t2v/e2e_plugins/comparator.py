"""wan_t2v model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class WanT2vDiffusionMediaGenerationComparator(DiffusionComparator):
    """wan_t2v local comparator for diffusion_media_generation."""

comparator = WanT2vDiffusionMediaGenerationComparator()
