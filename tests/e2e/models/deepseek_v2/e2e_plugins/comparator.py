"""deepseek_v2 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class DeepseekV2TextGenerationCausalComparator(TextComparator):
    """deepseek_v2 local comparator for text_generation_causal."""

comparator = DeepseekV2TextGenerationCausalComparator()
