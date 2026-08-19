# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""elf_flow model-owned E2E reference plugins."""

from __future__ import annotations

from .references.upstream_replay import ElfUpstreamReplayReference


class ElfFlowReference(ElfUpstreamReplayReference):
    """elf_flow local reference for official upstream JAX replay artifacts."""

reference = ElfFlowReference()
