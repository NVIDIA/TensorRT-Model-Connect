"""codegen model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class CodegenTextGenerationCausalComparator(TextComparator):
    """codegen local comparator for text_generation_causal."""

comparator = CodegenTextGenerationCausalComparator()
