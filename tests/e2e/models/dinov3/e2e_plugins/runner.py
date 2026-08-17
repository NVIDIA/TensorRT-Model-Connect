# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model-owned E2E runner plugin."""

from __future__ import annotations

from .runners.image_feature_extraction import ImageFeatureExtractionRunner


class Dinov3ImageFeatureExtractionRunner(ImageFeatureExtractionRunner):
    """DINOv3 runtime runner for image feature extraction."""


runner = Dinov3ImageFeatureExtractionRunner()
