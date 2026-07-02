# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sam model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.segmentation import PromptedSegmentationComparator


class SamPromptedSegmentationComparator(PromptedSegmentationComparator):
    """sam local comparator for prompted_segmentation."""

comparator = SamPromptedSegmentationComparator()
