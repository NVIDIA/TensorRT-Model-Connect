"""locateanything model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class LocateanythingVisionLanguageGenerationComparator(VisionLanguageComparator):
    """locateanything local comparator for vision_language_generation."""

comparator = LocateanythingVisionLanguageGenerationComparator()
