# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical checks for the family-owned TensorRT graph."""

from __future__ import annotations

import numpy as np
import pytest
import tensorrt as trt
import torch

from families.birefnet import graph


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_layer_norm_preserves_large_activation_features(dtype):
    # Swin stage 2 produces sparse activations beyond the FP16 square range.
    values = np.tile(np.linspace(-1, 1, 384), (1, 2, 1)).astype(dtype)
    values[0, 1, :2] = [-1318.0, 907.5]
    gamma = np.linspace(0.5, 1.5, 384).astype(dtype)
    beta = np.linspace(-0.1, 0.1, 384).astype(dtype)
    tensor_type = trt.float16 if dtype == np.float16 else trt.float32

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    source = network.add_input("source", tensor_type, values.shape)
    output = graph.add_layer_norm(network, source, gamma, beta, 1e-5, dtype=dtype)
    assert output.dtype == tensor_type
    output.name = "normalized"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    source_tensor = torch.from_numpy(values).cuda()
    actual = torch.empty_like(source_tensor)
    context.set_tensor_address("source", source_tensor.data_ptr())
    context.set_tensor_address("normalized", actual.data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    expected = torch.nn.functional.layer_norm(
        source_tensor,
        (384,),
        torch.from_numpy(gamma).cuda(),
        torch.from_numpy(beta).cuda(),
        1e-5,
    )
    tolerance = 2e-3 if dtype == np.float16 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
