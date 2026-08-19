# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""olmo2 model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class Olmo2TextGenerationCausalRunner(TextGenerationCausalRunner):
    """olmo2 local runner for text_generation_causal."""

runner = Olmo2TextGenerationCausalRunner()
