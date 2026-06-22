"""bart model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class BartTextGenerationCausalComparator(TextComparator):
    """bart local comparator for text_generation_causal."""

comparator = BartTextGenerationCausalComparator()
