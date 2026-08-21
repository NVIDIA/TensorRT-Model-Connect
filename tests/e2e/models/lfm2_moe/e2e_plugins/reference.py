# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-MoE model-owned E2E reference registration."""

from __future__ import annotations

from .references.hf_transformers import Lfm2MoeHfTransformersReference


class Lfm2MoeReference(Lfm2MoeHfTransformersReference):
    """Pinned LFM2-MoE Transformers oracle."""


reference = Lfm2MoeReference()
