"""segformer model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.segmentation import SegmentationRunner


class SegformerSegmentationRunner(SegmentationRunner):
    """segformer local runner for segmentation."""

runner = SegformerSegmentationRunner()
