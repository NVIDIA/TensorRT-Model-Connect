# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""phi4_multimodal model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Phi4MultimodalHfTransformersReference(HfTransformersReference):
    """phi4_multimodal local reference for hf_transformers."""

reference = Phi4MultimodalHfTransformersReference()
