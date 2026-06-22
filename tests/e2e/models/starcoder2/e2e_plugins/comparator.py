"""starcoder2 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class Starcoder2TextGenerationCausalComparator(TextComparator):
    """starcoder2 local comparator for text_generation_causal."""

comparator = Starcoder2TextGenerationCausalComparator()
