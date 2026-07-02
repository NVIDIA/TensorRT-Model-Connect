# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_omni model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.omni import OmniComparator


class Qwen3OmniMultimodalComparator(OmniComparator):
    """qwen3_omni local comparator for omni_multimodal."""


comparator = Qwen3OmniMultimodalComparator()
