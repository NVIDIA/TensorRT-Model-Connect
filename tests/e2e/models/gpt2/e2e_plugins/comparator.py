"""gpt2 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class Gpt2TextGenerationCausalComparator(TextComparator):
    """gpt2 local comparator for text_generation_causal."""

comparator = Gpt2TextGenerationCausalComparator()
