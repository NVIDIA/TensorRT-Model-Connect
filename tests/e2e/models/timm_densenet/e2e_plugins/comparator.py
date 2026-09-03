# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_densenet model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.image_classification import ImageClassificationComparator


class TimmDensenetImageClassificationComparator(ImageClassificationComparator):
    """timm_densenet local comparator for image_classification."""

comparator = TimmDensenetImageClassificationComparator()
