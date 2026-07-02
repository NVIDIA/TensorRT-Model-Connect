# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nemotron_h model-owned E2E reference plugins."""

from __future__ import annotations

from .references.golden_snapshot import GoldenSnapshotReference


class NemotronHGoldenSnapshotReference(GoldenSnapshotReference):
    """nemotron_h local reference for golden_snapshot."""

reference = NemotronHGoldenSnapshotReference()
