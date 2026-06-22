"""stablelm model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class StablelmTextGenerationCausalComparator(TextComparator):
    """stablelm local comparator for text_generation_causal."""

comparator = StablelmTextGenerationCausalComparator()
