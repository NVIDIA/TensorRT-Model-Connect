# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""chronos_bolt model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.neural_operator import NeuralOperatorRunner


class ChronosBoltNeuralOperatorRunner(NeuralOperatorRunner):
    """chronos_bolt local runner for neural_operator."""

runner = ChronosBoltNeuralOperatorRunner()
