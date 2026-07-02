# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mistral model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class MistralTextGenerationCausalRunner(TextGenerationCausalRunner):
    """mistral local runner for text_generation_causal."""

runner = MistralTextGenerationCausalRunner()
