"""phi model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class PhiTextGenerationCausalComparator(TextComparator):
    """phi local comparator for text_generation_causal."""

comparator = PhiTextGenerationCausalComparator()
