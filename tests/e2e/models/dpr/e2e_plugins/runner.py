# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dpr model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class DprEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """dpr local runner for encoder_only_nlp."""

runner = DprEncoderOnlyNlpRunner()
