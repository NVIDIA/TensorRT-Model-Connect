# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial BF16 projection checks against an independent FP64 oracle."""

import json
import struct

import numpy as np
import pytest


def cancellation_fixture(bfloat16):
    random = np.random.default_rng(24371)
    half = random.normal(size=1024).astype(bfloat16).astype(np.float32)
    half[-1] = 129 / 128
    permutation = random.permutation(1024)
    hidden = np.concatenate((half, half[permutation])).reshape(1, -1).astype(bfloat16)
    first = random.normal(0, 0.01, size=(64, 1024)).astype(bfloat16).astype(np.float32)
    kernel = np.concatenate((first, -first[:, permutation]), axis=1)
    partner = 1024 + int(np.flatnonzero(permutation == 1023)[0])
    # Most terms cancel exactly. The surviving product lies on opposite sides
    # of a BF16 rounding midpoint, so dropping the BF16 boundary is observable.
    for row in range(64):
        sign = 1 if row % 4 < 2 else -1
        kernel[row, 1023] = sign
        kernel[row, partner] = -sign / (256 if row % 2 == 0 else 512)
    kernel = kernel.astype(bfloat16)
    exact = hidden.astype(np.float64) @ kernel.astype(np.float64).T
    expected = exact.astype(bfloat16)
    # BF16 operands are dyadic; paired terms cancel exactly in the independent
    # FP64 oracle, leaving a controlled near-midpoint projection value.
    assert np.max(np.abs(hidden.astype(np.float64)) @ np.abs(kernel.astype(np.float64)).T) < 256
    return hidden, kernel, expected


def checkpoint_weights(directory, kernel):
    from families.qwen3_next.weights import Weights

    raw = kernel.view(np.uint16).tobytes()
    header = json.dumps(
        {
            "projection.weight": {
                "dtype": "BF16",
                "shape": list(kernel.shape),
                "data_offsets": [0, len(raw)],
            }
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    (directory / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"projection.weight": "part.safetensors"}})
    )
    return Weights(directory)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("fp32_consumer", [False, True])
def test_dense_projection_accumulates_fp32_and_retains_bf16_boundary(tmp_path, fp32_consumer):
    trt = pytest.importorskip("tensorrt")
    bfloat16 = pytest.importorskip("ml_dtypes").bfloat16
    cuda = pytest.importorskip("cuda.bindings.runtime")
    status, count = cuda.cudaGetDeviceCount()
    if int(status) or count == 0:
        pytest.skip("requires a CUDA GPU for the BF16 TensorRT projection")
    from families.qwen3_next.graph import Graph
    from families.qwen3_next.tests.gpu_runner import run

    hidden, kernel, expected = cancellation_fixture(bfloat16)
    weights = checkpoint_weights(tmp_path, kernel)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, weights)
    value = network.add_input("hidden", trt.bfloat16, hidden.shape)
    result = graph.linear(value, "projection.weight")
    if fp32_consumer:
        # Match the mixer's BF16 projection followed by FP32 gate arithmetic.
        # Only this FP32 consumer is exposed, so an extra BF16 output cannot
        # accidentally force a precision boundary absent from production.
        result = graph.mul(graph.cast(result, trt.float32), graph.constant([[3.0]]))
        expected = expected.astype(np.float32) * np.float32(3.0)
    result.name = "result"
    network.mark_output(result)
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 * 1024**2)
    plan = builder.build_serialized_network(network, options)
    assert plan is not None
    weights.finish()
    actual = run(bytes(plan), {"hidden": hidden})["result"]
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))
