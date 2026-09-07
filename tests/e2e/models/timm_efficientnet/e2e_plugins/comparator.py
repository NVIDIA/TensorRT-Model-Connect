# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_efficientnet model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.image_classification import ImageClassificationComparator


class TimmEfficientnetImageClassificationComparator(ImageClassificationComparator):
    """timm_efficientnet local comparator for image_classification."""

comparator = TimmEfficientnetImageClassificationComparator()
