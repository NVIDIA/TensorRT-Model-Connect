"""sam model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.segmentation import PromptedSegmentationRunner


class SamPromptedSegmentationRunner(PromptedSegmentationRunner):
    """sam local runner for prompted_segmentation."""

runner = SamPromptedSegmentationRunner()
