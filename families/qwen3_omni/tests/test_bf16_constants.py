# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Qwen3-Omni BF16 TensorRT constants."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from .. import graph_ops  # noqa: E402
from ..talker_builder import _build_text_projection_plan  # noqa: E402


def test_fp32_input_with_bf16_target_uses_exact_fp32_carrier() -> None:
    values = np.array([1.001, -0.3333, 17.0625], dtype=np.float32)
    carrier = graph_ops._constant_carrier(values, ml_dtypes.bfloat16)
    expected = values.astype(ml_dtypes.bfloat16).astype(np.float32)

    assert carrier.dtype == np.float32
    assert carrier.flags.c_contiguous
    np.testing.assert_array_equal(carrier, expected)
    assert not np.array_equal(carrier, values)


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


@pytest.mark.gpu
@pytest.mark.trt
def test_text_projection_matches_fused_bf16_silu_boundaries() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TensorRT execution")

    rng = np.random.default_rng(7)
    embedding = rng.normal(0.0, 0.1, (64, 16)).astype(ml_dtypes.bfloat16)
    projection = {
        "fc1": rng.normal(0.0, 0.1, (16, 32)).astype(ml_dtypes.bfloat16),
        "fc1_bias": rng.normal(0.0, 0.1, (32,)).astype(ml_dtypes.bfloat16),
        "fc2": rng.normal(0.0, 0.1, (32, 16)).astype(ml_dtypes.bfloat16),
        "fc2_bias": rng.normal(0.0, 0.1, (16,)).astype(ml_dtypes.bfloat16),
    }
    plan = _build_text_projection_plan(
        embedding,
        projection,
        max_tokens=8,
        precision="bf16",
        verbose=False,
    )

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    token_ids = torch.tensor([3, 5, 8, 13, 21, 34, 55], device="cuda", dtype=torch.int32)
    assert context.set_input_shape("token_id", (token_ids.numel(),))
    actual = torch.empty((token_ids.numel(), 16), device="cuda", dtype=torch.float32)
    assert context.set_tensor_address("token_id", token_ids.data_ptr())
    assert context.set_tensor_address("embeddings", actual.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    def bf16_tensor(value: np.ndarray):
        return torch.from_numpy(value.astype(np.float32)).to(device="cuda", dtype=torch.bfloat16)

    selected = bf16_tensor(embedding)[token_ids.to(torch.int64)]
    hidden = torch.nn.functional.linear(
        selected,
        bf16_tensor(projection["fc1"]).T,
        bf16_tensor(projection["fc1_bias"]),
    )
    hidden = torch.nn.functional.silu(hidden)
    expected = torch.nn.functional.linear(
        hidden,
        bf16_tensor(projection["fc2"]).T,
        bf16_tensor(projection["fc2_bias"]),
    )
    torch.testing.assert_close(actual, expected.float(), rtol=0.0, atol=0.0)
