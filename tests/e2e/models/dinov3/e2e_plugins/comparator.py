# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model-owned comparator plugin."""

from __future__ import annotations

from .comparators.image_feature_extraction import ImageFeatureExtractionComparator


class Dinov3ImageFeatureExtractionComparator(ImageFeatureExtractionComparator):
    """DINOv3 semantic image-feature comparator."""


comparator = Dinov3ImageFeatureExtractionComparator()
