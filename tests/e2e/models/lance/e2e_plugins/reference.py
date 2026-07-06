# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lance model-owned E2E reference plugins."""

from __future__ import annotations

from .references.golden_snapshot import GoldenSnapshotReference
from .references.hf_transformers import HfTransformersReference


class LanceGoldenSnapshotReference(GoldenSnapshotReference):
    """lance local reference for an upstream eager snapshot."""


class LanceHfTransformersReference(HfTransformersReference):
    """lance local reference for hf_transformers."""

reference = [
    LanceGoldenSnapshotReference(),
    LanceHfTransformersReference(),
]
