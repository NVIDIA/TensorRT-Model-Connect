# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""locateanything model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.vision_language import VisionLanguageRunner


class LocateanythingVisionLanguageGenerationRunner(VisionLanguageRunner):
    """locateanything local runner for vision_language_generation."""

runner = LocateanythingVisionLanguageGenerationRunner()
