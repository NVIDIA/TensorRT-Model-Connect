"""phi4_multimodal model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class Phi4MultimodalVisionLanguageGenerationComparator(VisionLanguageComparator):
    """phi4_multimodal local comparator for vision_language_generation."""

comparator = Phi4MultimodalVisionLanguageGenerationComparator()
