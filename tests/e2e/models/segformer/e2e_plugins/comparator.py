# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""segformer model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.segmentation import SegmentationComparator


class SegformerSegmentationComparator(SegmentationComparator):
    """segformer local comparator for segmentation."""

comparator = SegformerSegmentationComparator()
