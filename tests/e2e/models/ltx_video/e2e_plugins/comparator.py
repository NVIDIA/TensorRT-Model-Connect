"""ltx_video model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class LtxVideoDiffusionMediaGenerationComparator(DiffusionComparator):
    """ltx_video local comparator for diffusion_media_generation."""

comparator = LtxVideoDiffusionMediaGenerationComparator()
