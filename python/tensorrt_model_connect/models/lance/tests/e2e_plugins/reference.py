# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lance model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference
from .references.lance_official import LanceOfficialReference


class LancePinnedOfficialReference(LanceOfficialReference):
    """Pinned upstream Lance x2t_image reference."""


class LanceHfTransformersReference(HfTransformersReference):
    """lance local reference for hf_transformers."""

reference = [
    LancePinnedOfficialReference(),
    LanceHfTransformersReference(),
]
