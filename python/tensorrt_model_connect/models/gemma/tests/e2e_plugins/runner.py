# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gemma model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class GemmaTextGenerationCausalRunner(TextGenerationCausalRunner):
    """gemma local runner for text_generation_causal."""

runner = GemmaTextGenerationCausalRunner()
