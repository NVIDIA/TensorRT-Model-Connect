# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary family-owned normalization precision contracts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Canary builder imports")

from families.canary import model  # noqa: E402


def test_norm_epsilon_uses_regular_tensor_for_full_fp32_build() -> None:
    regular_eps = object()

    assert model._select_norm_eps(regular_eps, None, np.float32) is regular_eps


def test_norm_epsilon_uses_promoted_tensor_for_mixed_precision() -> None:
    regular_eps = object()
    promoted_eps = object()

    assert model._select_norm_eps(regular_eps, promoted_eps, np.float32) is promoted_eps
