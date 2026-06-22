"""mamba model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class MambaTextGenerationCausalComparator(TextComparator):
    """mamba local comparator for text_generation_causal."""

comparator = MambaTextGenerationCausalComparator()
