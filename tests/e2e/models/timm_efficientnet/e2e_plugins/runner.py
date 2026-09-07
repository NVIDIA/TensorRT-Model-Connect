# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_efficientnet model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.image_classification import ImageClassificationRunner


class TimmEfficientnetImageClassificationRunner(ImageClassificationRunner):
    """timm_efficientnet local runner for image_classification."""

runner = TimmEfficientnetImageClassificationRunner()
