# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timesfm model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.neural_operator import NeuralOperatorRunner


class TimesfmNeuralOperatorRunner(NeuralOperatorRunner):
    """timesfm local runner for neural_operator."""

runner = TimesfmNeuralOperatorRunner()
