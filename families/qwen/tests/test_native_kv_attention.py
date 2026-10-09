# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Masked native-KV attention ignores whatever unwritten cache rows contain."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for native KV attention")
cudart = pytest.importorskip("cuda.bindings.runtime", reason="cuda-python is required")

from families.qwen.native_kv_attention_builder import (  # noqa: E402
    add_active_prefix_causal_masks,
    add_explicit_masked_grouped_query_attention,
)

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

HEADS, KV_HEADS, HEAD_DIM, CAPACITY = 4, 2, 8, 8


def _check(result):
    error = result[0] if isinstance(result, tuple) else result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA call failed: {error}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def _build(dtype: trt.DataType, query_length: int) -> bytes:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    query = network.add_input("query", dtype, (1, HEADS, query_length, HEAD_DIM))
    key = network.add_input("key", dtype, (1, KV_HEADS, CAPACITY, HEAD_DIM))
    value = network.add_input("value", dtype, (1, KV_HEADS, CAPACITY, HEAD_DIM))
    rows = network.add_input("rows", trt.float32, (query_length, 1))
    write_index = network.add_input("write_index", trt.int32, (1,))
    length = network.add_input("length", trt.int32, (1,))
    masks = add_active_prefix_causal_masks(network, rows, write_index, length, CAPACITY)
    context = add_explicit_masked_grouped_query_attention(
        network,
        query,
        key,
        value,
        masks,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    context = network.add_cast(context, trt.float32).get_output(0)
    context.name = "context"
    network.mark_output(context)
    config = builder.create_builder_config()
    # Compare float32 against the float64 reference without TF32 matmul rounding.
    config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    return bytes(plan)


def _run(plan: bytes, inputs: dict[str, np.ndarray], output_shape: tuple[int, ...]) -> np.ndarray:
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    stream = _check(cudart.cudaStreamCreate())
    output = np.empty(output_shape, dtype=np.float32)
    buffers = []
    try:
        for name, array in [*inputs.items(), ("context", output)]:
            pointer = _check(cudart.cudaMalloc(array.nbytes))
            buffers.append(pointer)
            context.set_tensor_address(name, pointer)
            if name != "context":
                _check(cudart.cudaMemcpy(
                    pointer, array.ctypes.data, array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ))
        assert context.execute_async_v3(stream)
        _check(cudart.cudaMemcpyAsync(
            output.ctypes.data, buffers[-1], output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
        ))
        _check(cudart.cudaStreamSynchronize(stream))
    finally:
        for pointer in buffers:
            _check(cudart.cudaFree(pointer))
        _check(cudart.cudaStreamDestroy(stream))
    return output


def _reference(query, key, value, write_index: int, length: int) -> np.ndarray:
    groups = HEADS // KV_HEADS
    query_length = query.shape[2]
    result = np.zeros((1, HEADS, query_length, HEAD_DIM), dtype=np.float64)
    for head in range(HEADS):
        keys = key[0, head // groups].astype(np.float64)
        values = value[0, head // groups].astype(np.float64)
        for row in range(query_length):
            visible = min(write_index + row + 1, length)
            scores = keys[:visible] @ query[0, head, row].astype(np.float64) / np.sqrt(HEAD_DIM)
            weights = np.exp(scores - scores.max())
            result[0, head, row] = (weights / weights.sum()) @ values[:visible]
    return result


@pytest.mark.parametrize("dtype", [trt.float32, trt.float16])
@pytest.mark.parametrize(
    ("query_length", "write_index", "length"),
    [(1, 4, 5), (3, 2, 5)],
    ids=["decode", "prefill-chunk"],
)
def test_unwritten_cache_rows_cannot_reach_the_context(dtype, query_length, write_index, length):
    numpy_dtype = np.float32 if dtype == trt.float32 else np.float16
    rng = np.random.default_rng(7)
    query = rng.standard_normal((1, HEADS, query_length, HEAD_DIM)).astype(numpy_dtype)
    key = rng.standard_normal((1, KV_HEADS, CAPACITY, HEAD_DIM)).astype(numpy_dtype)
    value = rng.standard_normal((1, KV_HEADS, CAPACITY, HEAD_DIM)).astype(numpy_dtype)
    # Rows at or beyond the active length were never written by this request.
    key[:, :, length:] = np.nan
    key[:, :, length:, 0] = np.inf
    value[:, :, length:] = np.nan
    value[:, :, length:, 1] = -np.inf

    actual = _run(
        _build(dtype, query_length),
        {
            "query": query,
            "key": key,
            "value": value,
            "rows": np.zeros((query_length, 1), dtype=np.float32),
            "write_index": np.array([write_index], dtype=np.int32),
            "length": np.array([length], dtype=np.int32),
        },
        (1, HEADS, query_length, HEAD_DIM),
    )

    assert np.isfinite(actual).all()
    tolerance = 1e-5 if dtype == trt.float32 else 5e-3
    np.testing.assert_allclose(
        actual, _reference(query, key, value, write_index, length), rtol=0, atol=tolerance
    )
