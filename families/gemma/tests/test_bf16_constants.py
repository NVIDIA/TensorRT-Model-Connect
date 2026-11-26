# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 weights reach TensorRT as BF16 constants with the cast's rounding."""

from __future__ import annotations

import gc

import ml_dtypes
import numpy as np
import pytest
import tensorrt as trt

from families.gemma import graph_ops


def test_bf16_bits_round_to_nearest_even_like_a_cast() -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal((37, 19)).astype(np.float16)
    expected = values.astype(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)

    bits = graph_ops.bf16_bits(values)

    assert bits.dtype == np.uint16 and bits.shape == values.shape
    np.testing.assert_array_equal(bits, expected)
    # Ties go to the even mantissa: 1 + 2^-8 rounds down, 1 + 3 * 2^-8 rounds up.
    ties = np.array([1.00390625, 1.01171875], dtype=np.float32)
    np.testing.assert_array_equal(graph_ops.bf16_bits(ties), [0x3F80, 0x3F82])


@pytest.mark.gpu
@pytest.mark.trt
def test_bf16_matmul_constant_matches_the_cast_form_and_lives_with_the_network() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(1)
    weights = rng.standard_normal((32, 48)).astype(np.float16)
    lhs = (
        torch.from_numpy(rng.standard_normal((4, 32)).astype(np.float32)).to(torch.bfloat16).cuda()
    )
    logger = trt.Logger(trt.Logger.ERROR)

    def run(direct: bool) -> np.ndarray:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        inp = network.add_input("x", trt.bfloat16, (4, 32))
        if direct:
            out = graph_ops.add_matmul_rhs_constant(network, inp, 32, 48, weights, dtype=np.float16)
            assert len(graph_ops._BF16_DATA[network]) == 1
        else:
            rhs = graph_ops.add_constant(network, (32, 48), weights, dtype=np.float16)
            rhs = network.add_cast(rhs, trt.bfloat16).get_output(0)
            mm = network.add_matrix_multiply(
                inp, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
            )
            out = mm.get_output(0)
        out.name = "y"
        network.mark_output(out)
        plan = builder.build_serialized_network(network, builder.create_builder_config())
        engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        result = torch.empty((4, 48), dtype=torch.bfloat16, device="cuda")
        context.set_tensor_address("x", lhs.data_ptr())
        context.set_tensor_address("y", result.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        del network
        gc.collect()
        return result.float().cpu().numpy()

    direct = run(True)
    assert len(graph_ops._BF16_DATA) == 0  # released with its network
    # Same weights, same graph after TensorRT folds the cast: only tactic choice may differ.
    np.testing.assert_allclose(direct, run(False), rtol=1e-2, atol=1e-2)
