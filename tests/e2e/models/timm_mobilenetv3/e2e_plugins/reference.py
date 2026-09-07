# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_mobilenetv3 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class TimmMobilenetv3HfTransformersReference(HfTransformersReference):
    """timm_mobilenetv3 local reference for hf_transformers."""

reference = TimmMobilenetv3HfTransformersReference()
