# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""patchtsmixer model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.neural_operator import NeuralOperatorComparator


class PatchtsmixerNeuralOperatorComparator(NeuralOperatorComparator):
    """patchtsmixer local comparator for neural_operator."""

comparator = PatchtsmixerNeuralOperatorComparator()
