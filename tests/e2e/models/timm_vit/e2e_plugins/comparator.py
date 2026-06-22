"""timm_vit model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.image_classification import ImageClassificationComparator


class TimmVitImageClassificationComparator(ImageClassificationComparator):
    """timm_vit local comparator for image_classification."""

comparator = TimmVitImageClassificationComparator()
