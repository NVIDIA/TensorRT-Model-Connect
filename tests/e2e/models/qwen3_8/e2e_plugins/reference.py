# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_8 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class Qwen38HfTransformersReference(HfTransformersReference):
    """qwen3_8 local reference for hf_transformers."""

reference = Qwen38HfTransformersReference()
