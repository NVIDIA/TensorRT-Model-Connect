"""xglm model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class XglmTextGenerationCausalComparator(TextComparator):
    """xglm local comparator for text_generation_causal."""

comparator = XglmTextGenerationCausalComparator()
