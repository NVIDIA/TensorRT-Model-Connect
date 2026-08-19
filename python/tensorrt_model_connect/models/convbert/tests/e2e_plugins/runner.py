# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""convbert model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class ConvbertEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """convbert local runner for encoder_only_nlp."""

runner = ConvbertEncoderOnlyNlpRunner()
