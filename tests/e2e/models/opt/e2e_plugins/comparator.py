"""opt model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class OptTextGenerationCausalComparator(TextComparator):
    """opt local comparator for text_generation_causal."""

comparator = OptTextGenerationCausalComparator()
