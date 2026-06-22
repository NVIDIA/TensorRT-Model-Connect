"""pixart model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class PixartDiffusionMediaGenerationComparator(DiffusionComparator):
    """pixart local comparator for diffusion_media_generation."""

comparator = PixartDiffusionMediaGenerationComparator()
