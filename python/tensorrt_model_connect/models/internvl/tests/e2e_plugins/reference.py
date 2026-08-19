# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""internvl model-owned E2E reference plugins."""

from __future__ import annotations

from .references.golden_snapshot import GoldenSnapshotReference
from .references.hf_transformers import HfTransformersReference


class InternvlGoldenSnapshotReference(GoldenSnapshotReference):
    """internvl local reference for golden_snapshot."""


class InternvlHfTransformersReference(HfTransformersReference):
    """internvl local reference for hf_transformers."""

reference = [
    InternvlGoldenSnapshotReference(),
    InternvlHfTransformersReference(),
]
