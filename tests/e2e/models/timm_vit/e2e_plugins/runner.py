"""timm_vit model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.image_classification import ImageClassificationRunner


class TimmVitImageClassificationRunner(ImageClassificationRunner):
    """timm_vit local runner for image_classification."""

runner = TimmVitImageClassificationRunner()
