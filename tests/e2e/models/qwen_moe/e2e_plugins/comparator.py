"""qwen_moe model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class QwenMoeTextGenerationCausalComparator(TextComparator):
    """qwen_moe local comparator for text_generation_causal."""

comparator = QwenMoeTextGenerationCausalComparator()
