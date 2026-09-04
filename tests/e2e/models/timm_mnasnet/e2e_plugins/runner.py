# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_mnasnet model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.image_classification import ImageClassificationRunner


class TimmMnasnetImageClassificationRunner(ImageClassificationRunner):
    """timm_mnasnet local runner for image_classification."""

runner = TimmMnasnetImageClassificationRunner()
