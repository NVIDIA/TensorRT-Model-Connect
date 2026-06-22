"""gpt_oss model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class GptOssTextGenerationCausalComparator(TextComparator):
    """gpt_oss local comparator for text_generation_causal."""

comparator = GptOssTextGenerationCausalComparator()
