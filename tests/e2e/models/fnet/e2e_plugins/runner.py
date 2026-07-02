# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fnet model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class FnetEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """fnet local runner for encoder_only_nlp."""

runner = FnetEncoderOnlyNlpRunner()
