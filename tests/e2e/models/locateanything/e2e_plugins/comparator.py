# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""locateanything model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.vision_language import VisionLanguageComparator


class LocateanythingVisionLanguageGenerationComparator(VisionLanguageComparator):
    """locateanything local comparator for vision_language_generation."""

comparator = LocateanythingVisionLanguageGenerationComparator()
