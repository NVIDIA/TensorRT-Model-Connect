# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""m2m_100 model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class M2m100HfTransformersReference(HfTransformersReference):
    """m2m_100 local reference for hf_transformers."""

reference = M2m100HfTransformersReference()
