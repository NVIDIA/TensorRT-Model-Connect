# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""internvl model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class InternvlVisionLanguageGenerationComparator(VisionLanguageComparator):
    """internvl local comparator for vision_language_generation."""

comparator = InternvlVisionLanguageGenerationComparator()
