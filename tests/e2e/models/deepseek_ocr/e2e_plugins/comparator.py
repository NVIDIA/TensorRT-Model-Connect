"""deepseek_ocr model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class DeepseekOcrVisionLanguageGenerationComparator(VisionLanguageComparator):
    """deepseek_ocr local comparator for vision_language_generation."""

comparator = DeepseekOcrVisionLanguageGenerationComparator()
