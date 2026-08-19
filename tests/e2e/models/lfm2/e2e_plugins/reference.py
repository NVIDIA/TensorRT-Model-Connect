# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2 model-owned E2E reference registration."""

from __future__ import annotations

from .references.hf_transformers import Lfm2HfTransformersReference


class Lfm2Reference(Lfm2HfTransformersReference):
    """Pinned dense-LFM2 Transformers oracle."""


reference = Lfm2Reference()
