"""z_image model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class ZImageDiffusionMediaGenerationComparator(DiffusionComparator):
    """z_image local comparator for diffusion_media_generation."""

comparator = ZImageDiffusionMediaGenerationComparator()
