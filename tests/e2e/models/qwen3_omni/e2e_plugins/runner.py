# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_omni model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.omni import OmniMultimodalRunner


class Qwen3OmniMultimodalRunner(OmniMultimodalRunner):
    """qwen3_omni local runner for omni_multimodal."""

    @property
    def strategy_name(self) -> str:
        return "omni_multimodal"


runner = Qwen3OmniMultimodalRunner()
