"""mistral model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class MistralTextGenerationCausalComparator(TextComparator):
    """mistral local comparator for text_generation_causal."""

comparator = MistralTextGenerationCausalComparator()
