# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_regnet model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.image_classification import ImageClassificationRunner


class TimmRegnetImageClassificationRunner(ImageClassificationRunner):
    """timm_regnet local runner for image_classification."""

runner = TimmRegnetImageClassificationRunner()
