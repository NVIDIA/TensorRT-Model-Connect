"""qwen_vl model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class QwenVlVisionLanguageGenerationComparator(VisionLanguageComparator):
    """qwen_vl local comparator for vision_language_generation."""

comparator = QwenVlVisionLanguageGenerationComparator()
