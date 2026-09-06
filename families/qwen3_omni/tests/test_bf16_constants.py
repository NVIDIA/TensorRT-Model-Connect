# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Qwen3-Omni BF16 TensorRT constants."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from .. import graph_ops  # noqa: E402


def test_fp32_input_with_bf16_target_uses_exact_fp32_carrier() -> None:
    values = np.array([1.001, -0.3333, 17.0625], dtype=np.float32)
    carrier = graph_ops._constant_carrier(values, ml_dtypes.bfloat16)
    expected = values.astype(ml_dtypes.bfloat16).astype(np.float32)

    assert carrier.dtype == np.float32
    assert carrier.flags.c_contiguous
    np.testing.assert_array_equal(carrier, expected)
    assert not np.array_equal(carrier, values)


def test_thinker_rope_table_matches_hf_float32_rounding() -> None:
    cosine = graph_ops.make_rope_table_half_dim(256, 128, 1_000_000.0, True)
    sine = graph_ops.make_rope_table_half_dim(256, 128, 1_000_000.0, False)

    assert cosine.shape == sine.shape == (256, 64)
    assert cosine.dtype == sine.dtype == np.float32
    assert cosine.astype(ml_dtypes.bfloat16)[64, 7] == ml_dtypes.bfloat16(0.01409912109375)
    assert sine.astype(ml_dtypes.bfloat16)[132, 9] == ml_dtypes.bfloat16(0.06640625)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("values_dtype", [np.float32, ml_dtypes.bfloat16])
def test_bf16_target_serializes_in_strongly_typed_network(values_dtype) -> None:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    input_tensor = network.add_input("input", trt.bfloat16, (3,))
    constant = graph_ops.add_constant(
        network,
        (3,),
        np.array([1.001, -0.3333, 17.0625], dtype=values_dtype),
        dtype=ml_dtypes.bfloat16,
    )
    assert constant.dtype == trt.float32
    constant_bf16 = network.add_cast(constant, trt.bfloat16).get_output(0)
    output = network.add_elementwise(
        input_tensor, constant_bf16, trt.ElementWiseOperation.SUM
    ).get_output(0)
    output.name = "output"
    network.mark_output(output)

    plan = builder.build_serialized_network(network, build_config)
    assert plan is not None
    assert bytes(plan)
