# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder optimization levels of the single-GPU Phi-MoE engines."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Phi-MoE builder tests")

from families.phi_moe import utils  # noqa: E402


def test_family_builder_optimization_level_is_the_tensorrt_default():
    assert utils.BUILDER_OPTIMIZATION_LEVEL == 3


def test_builder_context_uses_the_family_optimization_level():
    context = utils.create_builder_context(verbose=False)
    assert context.config.builder_optimization_level == utils.BUILDER_OPTIMIZATION_LEVEL


def test_decode_level_is_the_family_level_for_unquantized_builds():
    assert utils.decode_builder_optimization_level(False) == utils.BUILDER_OPTIMIZATION_LEVEL


def test_fp8_decode_level_stays_at_one():
    assert utils.decode_builder_optimization_level(True) == 1
