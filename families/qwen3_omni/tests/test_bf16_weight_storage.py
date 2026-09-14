# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 checkpoint constants retain compact storage and their original values."""

from __future__ import annotations

import gc

import ml_dtypes
import numpy as np
import pytest
import tensorrt as trt

from .. import graph_ops


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("values_dtype", [np.float32, ml_dtypes.bfloat16])
def test_bf16_constants_keep_compact_storage_and_survive_collection(values_dtype) -> None:
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    inp = network.add_input("input", trt.bfloat16, (3,))
    values = np.array([1.001, 0.0, -0.3333, 0.0, 17.0625, 0.0], dtype=values_dtype)[::2]
    expected = values.astype(ml_dtypes.bfloat16).astype(np.float32)
    constant = graph_ops.add_constant(network, (3,), values, dtype=ml_dtypes.bfloat16)
    layer = network.get_layer(0)

    assert network.num_layers == 1
    assert layer.type == trt.LayerType.CONSTANT
    assert tuple(constant.shape) == (3,)
    assert constant.dtype == trt.bfloat16

    # Exercise the lifetime of a converted, noncontiguous temporary array.
    del values
    gc.collect()
    output = network.add_elementwise(inp, constant, trt.ElementWiseOperation.SUM).get_output(0)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    inputs = torch.zeros(3, dtype=torch.bfloat16, device="cuda")
    outputs = torch.empty_like(inputs)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    assert context.set_tensor_address("input", inputs.data_ptr())
    assert context.set_tensor_address("output", outputs.data_ptr())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    np.testing.assert_array_equal(outputs.float().cpu().numpy(), expected)
