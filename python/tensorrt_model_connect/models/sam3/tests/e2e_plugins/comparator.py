# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sam3 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.segmentation import PromptedSegmentationComparator


class Sam3PromptedSegmentationComparator(PromptedSegmentationComparator):
    """Sam3 local comparator for prompted_segmentation."""

comparator = Sam3PromptedSegmentationComparator()
