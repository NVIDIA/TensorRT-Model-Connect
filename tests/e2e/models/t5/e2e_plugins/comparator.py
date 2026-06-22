"""t5 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class T5TextGenerationCausalComparator(TextComparator):
    """t5 local comparator for text_generation_causal."""

comparator = T5TextGenerationCausalComparator()
