"""internvl model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class InternvlVisionLanguageGenerationComparator(VisionLanguageComparator):
    """internvl local comparator for vision_language_generation."""

comparator = InternvlVisionLanguageGenerationComparator()
