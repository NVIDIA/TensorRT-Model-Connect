"""nemotron_labs_diffusion model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class NemotronLabsDiffusionTextGenerationCausalComparator(TextComparator):
    """nemotron_labs_diffusion local comparator for text_generation_causal."""

comparator = NemotronLabsDiffusionTextGenerationCausalComparator()
