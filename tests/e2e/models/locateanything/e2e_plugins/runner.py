"""locateanything model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.vision_language import VisionLanguageRunner


class LocateanythingVisionLanguageGenerationRunner(VisionLanguageRunner):
    """locateanything local runner for vision_language_generation."""

runner = LocateanythingVisionLanguageGenerationRunner()
