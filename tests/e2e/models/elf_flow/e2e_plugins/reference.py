# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""elf_flow model-owned E2E reference plugins."""

from __future__ import annotations

from .references.invariant_only import InvariantOnlyReference


class ElfFlowInvariantOnlyReference(InvariantOnlyReference):
    """elf_flow local reference for invariant_only."""

reference = ElfFlowInvariantOnlyReference()
