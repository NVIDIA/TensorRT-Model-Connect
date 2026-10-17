# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and memory regressions for variable-resolution FP32 vision attention."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.qwen_vl.graph_ops import (  # noqa: E402
    _add_chunked_vision_attention,
    add_dynamic_self_attention_with_rope,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

HEADS, HEAD_DIM, MAX_PATCHES = 16, 80, 10208


@pytest.fixture(scope="module")
def engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors = [network.add_input(name, trt.float32, (-1, HEADS * HEAD_DIM))
               for name in ("q", "k", "v")]
    output = _add_chunked_vision_attention(
        network, *tensors, num_heads=HEADS, head_dim=HEAD_DIM)
    output.name = "context"
    network.mark_output(output)
    profile = builder.create_optimization_profile()
    for tensor in tensors:
        profile.set_shape(tensor.name, (1, HEADS * HEAD_DIM),
                          (257, HEADS * HEAD_DIM), (MAX_PATCHES, HEADS * HEAD_DIM))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 << 20)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "FP32 dynamic vision attention must build"
    runtime = trt.Runtime(logger)
    built = runtime.deserialize_cuda_engine(plan)
    assert built is not None
    yield built


def test_large_image_attention_avoids_dense_score_buffers(engine):
    # Dense scores + probabilities at this real failing image size use 13.3 GB.
    # Bound the selected runtime buffers, including tactic workspace. The
    # builder cap is higher because its loop tactic requires a separate budget.
    assert engine.device_memory_size_v2 < 512 << 20


@pytest.mark.parametrize("patches", [1, 31, 32, 33, 127, 128, 129, 257, MAX_PATCHES])
def test_dynamic_attention_matches_independent_softmax_across_query_boundaries(engine, patches):
    rng = np.random.default_rng(20261017 + patches)
    inputs = {name: rng.standard_normal((patches, HEADS * HEAD_DIM)).astype(np.float32)
              for name in ("q", "k", "v")}
    context = engine.create_execution_context()
    assert context is not None
    buffers = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
    for name, tensor in buffers.items():
        assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty((patches, HEADS * HEAD_DIM), device="cuda", dtype=torch.float32)
    assert tuple(context.get_tensor_shape("context")) == tuple(output.shape)
    assert context.set_tensor_address("context", output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = output.cpu().numpy()

    # Include both sides of chunk boundaries and the last non-padded row. The
    # reference keeps every key, but samples queries to avoid a dense N x N copy.
    indices = sorted({i for i in [0, 1, 30, 31, 32, 33, 126, 127, 128, 129, 255, 256, patches - 1]
                      if 0 <= i < patches})
    q, k, v = [inputs[name].reshape(patches, HEADS, HEAD_DIM).transpose(1, 0, 2)
               .astype(np.float64) for name in ("q", "k", "v")]
    scores = (q[:, indices] @ k.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = (probabilities @ v).transpose(1, 0, 2).reshape(len(indices), -1)
    np.testing.assert_allclose(actual[indices], expected, rtol=2e-5, atol=2e-5)


def test_chained_dynamic_attention_preserves_nonuniform_scores_after_rope():
    """Exercise loop integration with computed Q/K, not just engine inputs."""
    width, heads, dimension, patches = 64, 4, 16, 12
    rng = np.random.default_rng(20261017)
    weights = [tuple((rng.standard_normal((width, width)) / np.sqrt(width)).astype(np.float32)
                     for _ in range(4)) for _ in range(2)]
    values = rng.standard_normal((patches, width)).astype(np.float32)
    angles = (np.arange(patches)[:, None] *
              (10000.0 ** (-np.arange(0, dimension, 2) / dimension)))
    cosine, sine = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    expected = values.astype(np.float64)
    for wq, wk, wv, wo in weights:
        q, k, v = [(expected @ weight).reshape(patches, heads, dimension).transpose(1, 0, 2)
                   for weight in (wq, wk, wv)]

        def rotate(tensor):
            first, second = np.split(tensor, 2, axis=-1)
            return np.concatenate((first * cosine - second * sine,
                                   second * cosine + first * sine), axis=-1)

        scores = rotate(q) @ rotate(k).transpose(0, 2, 1) / np.sqrt(dimension)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = (probabilities @ v).transpose(1, 0, 2).reshape(patches, width)
        expected = expected + context @ wo

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inputs = [network.add_input("hidden", trt.float32, (-1, width)),
              network.add_input("cosine", trt.float32, (-1, dimension // 2)),
              network.add_input("sine", trt.float32, (-1, dimension // 2))]
    hidden = inputs[0]
    for wq, wk, wv, wo in weights:
        attended = add_dynamic_self_attention_with_rope(
            network, hidden, wq, wk, wv, wo, width, heads, inputs[1], inputs[2])
        hidden = network.add_elementwise(hidden, attended, trt.ElementWiseOperation.SUM).get_output(0)
    hidden.name = "output"
    network.mark_output(hidden)
    profile = builder.create_optimization_profile()
    for tensor, columns in zip(inputs, (width, dimension // 2, dimension // 2)):
        profile.set_shape(tensor.name, (1, columns), (12, columns), (129, columns))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 << 20)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    execution = engine.create_execution_context()
    assert execution is not None
    buffers = {tensor.name: torch.from_numpy(array).cuda()
               for tensor, array in zip(inputs, (values, cosine, sine))}
    for name, tensor in buffers.items():
        assert execution.set_input_shape(name, tuple(tensor.shape))
        assert execution.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty((patches, width), device="cuda", dtype=torch.float32)
    assert execution.set_tensor_address("output", output.data_ptr())
    assert execution.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=2e-5, atol=2e-5)
