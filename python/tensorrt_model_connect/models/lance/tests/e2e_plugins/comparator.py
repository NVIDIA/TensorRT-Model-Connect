# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lance model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class LanceVisionLanguageGenerationComparator(VisionLanguageComparator):
    """lance local comparator for vision_language_generation."""

comparator = LanceVisionLanguageGenerationComparator()
