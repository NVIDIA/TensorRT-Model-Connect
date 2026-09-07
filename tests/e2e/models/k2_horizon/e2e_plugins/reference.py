# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon model-owned E2E reference registration."""

from __future__ import annotations

from .references.hf_transformers import K2HorizonHfTransformersReference


class K2HorizonReference(K2HorizonHfTransformersReference):
    """Pinned dense-K2-Horizon Transformers oracle."""


reference = K2HorizonReference()
