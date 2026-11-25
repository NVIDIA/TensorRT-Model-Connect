# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's TensorRT attention, built alone on the GPU: its operation order and fp16 numerics."""

from __future__ import annotations

import numpy as np
import pytest
import tensorrt as trt
import torch

from families.dpt import graph

HEAD_DIM = 64  # the checkpoint's; the attention scale is 1/8
_LOGGER = trt.Logger(trt.Logger.WARNING)


def _run_fp16(build, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Build ``build(network, *inputs)`` as a strongly typed fp16 engine, run it once, return float32."""
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors = [network.add_input(name, trt.float16, value.shape) for name, value in inputs.items()]
    output = graph.retype(network, build(network, *tensors), trt.float32)
    output.name = "output"
    network.mark_output(output)
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    plan = builder.build_serialized_network(network, settings)
    assert plan is not None
    engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    device = {name: torch.from_numpy(value).to("cuda", torch.float16) for name, value in inputs.items()}
    result = torch.full(tuple(output.shape), float("nan"), dtype=torch.float32, device="cuda")
    for name, tensor in device.items():
        context.set_tensor_address(name, tensor.data_ptr())
    context.set_tensor_address("output", result.data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return result.cpu().numpy()


def _hadamard(size: int) -> np.ndarray:
    matrix = np.ones((1, 1))
    while len(matrix) < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


@pytest.mark.gpu
@pytest.mark.trt
def test_attention_multiplies_the_query_before_the_score_product():
    if not torch.cuda.is_available():
        pytest.skip("a TensorRT network needs CUDA")
    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inputs = [network.add_input(name, trt.float16, (1, 2, 4, HEAD_DIM))
              for name in ("query", "key", "value")]
    graph.add_attention(network, *inputs, 1.0 / np.sqrt(HEAD_DIM))
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    products = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    (softmax,) = [layer for layer in layers if layer.type == trt.LayerType.SOFTMAX]
    (scaling,) = [layer for layer in layers if layer.type == trt.LayerType.ELEMENTWISE]
    scaling.__class__ = trt.IElementWiseLayer
    # Q * scale feeds the score product, whose output goes straight into the softmax.
    assert scaling.op == trt.ElementWiseOperation.PROD
    assert scaling.get_input(0).name == "query"
    assert products[0].get_input(0).name == scaling.get_output(0).name
    assert products[0].get_input(1).name == "key"
    assert softmax.get_input(0).name == products[0].get_output(0).name


@pytest.mark.gpu
@pytest.mark.trt
def test_fp16_attention_scales_the_query_before_the_scores_can_overflow():
    if not torch.cuda.is_available():
        pytest.skip("the attention engine needs CUDA")
    rng = np.random.default_rng(0)
    signs = _hadamard(HEAD_DIM)
    query = np.empty((1, 2, 4, HEAD_DIM))
    key = np.empty_like(query)
    # Head 0: orthogonal rows of +-40, so Q K^T is 0 or 64 * 40 * 40 = 102400, past fp16's
    # 65504, while the scaled score 12800 is exact in fp16. Queries 0 and 1 each match one
    # key (one-hot weights), queries 2 and 3 none (uniform weights).
    query[0, 0] = 40 * signs[[1, 2, 3, 5]]
    key[0, 0] = 40 * signs[[1, 2, 4, 6]]
    # Head 1: ordinary scores.
    query[0, 1] = rng.standard_normal((4, HEAD_DIM))
    key[0, 1] = rng.standard_normal((4, HEAD_DIM))
    value = rng.standard_normal((1, 2, 4, HEAD_DIM))
    inputs = {"query": query.astype(np.float16), "key": key.astype(np.float16),
              "value": value.astype(np.float16)}
    q, k, v = (inputs[name].astype(np.float64) for name in ("query", "key", "value"))
    unscaled = q @ k.swapaxes(-1, -2)
    assert unscaled.max() > np.finfo(np.float16).max > unscaled.max() / 8

    actual = _run_fp16(
        lambda network, *qkv: graph.add_attention(network, *qkv, 1.0 / np.sqrt(HEAD_DIM)), inputs)

    scores = unscaled / np.sqrt(HEAD_DIM)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = weights / weights.sum(axis=-1, keepdims=True) @ v
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)
