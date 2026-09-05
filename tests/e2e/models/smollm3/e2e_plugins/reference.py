# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""smollm3 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class SmolLM3HfTransformersReference(HfTransformersReference):
    """smollm3 local reference for hf_transformers."""


reference = [SmolLM3HfTransformersReference()]
