# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deepseek_ocr model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class DeepseekOcrVisionLanguageGenerationComparator(VisionLanguageComparator):
    """deepseek_ocr local comparator for vision_language_generation."""

comparator = DeepseekOcrVisionLanguageGenerationComparator()
