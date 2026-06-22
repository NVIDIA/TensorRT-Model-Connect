"""flux model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class FluxDiffusionMediaGenerationComparator(DiffusionComparator):
    """flux local comparator for diffusion_media_generation."""

comparator = FluxDiffusionMediaGenerationComparator()
