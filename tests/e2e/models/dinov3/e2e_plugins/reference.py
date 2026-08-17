# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model-owned reference plugin."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Dinov3HfTransformersReference(HfTransformersReference):
    """DINOv3 AutoImageProcessor + AutoModel reference."""


reference = Dinov3HfTransformersReference()
