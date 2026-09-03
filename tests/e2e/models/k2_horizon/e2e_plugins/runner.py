# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon model-owned runner registration."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class K2HorizonTextGenerationCausalRunner(TextGenerationCausalRunner):
    """K2-Horizon dense grouped-RMSNorm generation runner."""


runner = K2HorizonTextGenerationCausalRunner()
