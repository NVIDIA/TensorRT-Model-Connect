# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-MoE model-owned runner registration."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class Lfm2MoeTextGenerationCausalRunner(TextGenerationCausalRunner):
    """LFM2-MoE hybrid-convolution/attention generation runner."""


runner = Lfm2MoeTextGenerationCausalRunner()
