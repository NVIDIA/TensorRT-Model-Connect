# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""phi_moe model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class PhiMoeHfTransformersReference(HfTransformersReference):
    """phi_moe local reference for hf_transformers."""

reference = PhiMoeHfTransformersReference()
