# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""internlm model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class InternlmHfTransformersReference(HfTransformersReference):
    """internlm local reference for hf_transformers."""

reference = InternlmHfTransformersReference()
