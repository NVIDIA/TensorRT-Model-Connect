"""nemotron_h model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class NemotronHTextGenerationCausalComparator(TextComparator):
    """nemotron_h local comparator for text_generation_causal."""

comparator = NemotronHTextGenerationCausalComparator()
