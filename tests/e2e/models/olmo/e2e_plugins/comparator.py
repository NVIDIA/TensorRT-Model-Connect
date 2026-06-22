"""olmo model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class OlmoTextGenerationCausalComparator(TextComparator):
    """olmo local comparator for text_generation_causal."""

comparator = OlmoTextGenerationCausalComparator()
