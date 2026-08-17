# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model-owned reference plugin."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference
from .references.timm_dinov3 import TimmDinov3Reference


class Dinov3HfTransformersReference(HfTransformersReference):
    """DINOv3 AutoImageProcessor + AutoModel reference."""


class Dinov3TimmReference(TimmDinov3Reference):
    """Independent public timm DINOv3 reference used by secretless PR CI."""

    @property
    def backend_name(self) -> str:
        return "timm_dinov3"


reference = [Dinov3HfTransformersReference(), Dinov3TimmReference()]
