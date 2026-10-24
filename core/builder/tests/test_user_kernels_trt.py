# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Toy examples: the pattern ``a * b + c`` is replaced by a user kernel at engine build.

The user kernel computes ``a * b + c + 1``, so its output cannot be mistaken for the output of the
original ops. In the first test the graph is exactly the pattern. In the second test the pattern is
one sub structure of a larger graph, and only that part is replaced. The kernel is either CUDA
source or a cuTile kernel that the helper builds into a cubin for the GPU in the machine. A TensorRT
build that ignores the build option returns the original result, and the tests skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.user_kernels import UserKernel, user_kernels

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

N = 64
T = "!nvg.tensor<f32, {[1], stride_order = [0]}>"
PATTERN = f"""  nvg.graph @fma_pattern(%a: {T}, %b: {T}, %c: {T}) -> {T}
      attributes {{}} {{
    %m = nvg.mul(%a, %b) : ({T}, {T}) -> {T}
    %s = nvg.add(%m, %c) : ({T}, {T}) -> {T}
    nvg.graph_return %s : {T}
  }}"""
SOURCE = (
    'extern "C" __global__ void USER_KERNEL_ENTRY(const float *a, const float *b, const float *c, '
    "float *out, long long n) {\n"
    "  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i < n) out[i] = a[i] * b[i] + c[i] + 1.0f;\n"
    "}"
)


def _make_kernel(kind: str) -> UserKernel:
    """The ``a * b + c + 1`` kernel as CUDA source or as a cuTile kernel built into a cubin."""
    if kind == "cuda":
        return UserKernel("user_fma", PATTERN, SOURCE, block=128)
    pytest.importorskip("cuda.tile")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from .user_kernel_cutile_fma import fma_tile

    return UserKernel.from_cutile(
        "user_fma_tile", PATTERN, fma_tile, tile=128, num_arrays=4, dtype="float32"
    )


def _build_engine():
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    a, b, c = (network.add_input(name, trt.float32, (N,)) for name in "abc")
    prod = network.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)
    out = network.add_elementwise(prod, c, trt.ElementWiseOperation.SUM).get_output(0)
    out.name = "out"
    network.mark_output(out)
    blob = builder.build_serialized_network(network, builder.create_builder_config())
    assert blob is not None, "engine build failed"
    return trt.Runtime(logger).deserialize_cuda_engine(blob)


def _run(engine, inputs):
    context = engine.create_execution_context()
    out = torch.empty(N, dtype=torch.float32, device="cuda")
    for name, tensor in zip("abc", inputs):
        context.set_tensor_address(name, tensor.data_ptr())
    context.set_tensor_address("out", out.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out.cpu().numpy()


def test_pattern_replaced_by_user_kernel():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    rng = np.random.default_rng(0)
    host = [rng.standard_normal(N).astype(np.float32) for _ in range(3)]
    inputs = [torch.from_numpy(x).cuda() for x in host]
    original = host[0] * host[1] + host[2]

    baseline = _run(_build_engine(), inputs)
    np.testing.assert_allclose(baseline, original, rtol=1e-5, atol=1e-6)

    with user_kernels([UserKernel("user_fma", PATTERN, SOURCE, block=128)]):
        replaced = _run(_build_engine(), inputs)

    if np.allclose(replaced, baseline, rtol=1e-5, atol=1e-6):
        pytest.skip("this TensorRT build ignores the user kernel option")
    np.testing.assert_allclose(replaced, original + 1.0, rtol=1e-5, atol=1e-6)


R, C = 8, 64


def _build_larger_engine(weights):
    """matmul, (p * s + r), relu, reduce_sum; only ``p * s + r`` matches the pattern."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x, s, r = (network.add_input(name, trt.float32, (R, C)) for name in ("x", "s", "r"))
    w = network.add_constant((C, C), trt.Weights(weights)).get_output(0)
    m = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.NONE
    ).get_output(0)
    prod = network.add_elementwise(m, s, trt.ElementWiseOperation.PROD).get_output(0)
    u = network.add_elementwise(prod, r, trt.ElementWiseOperation.SUM).get_output(0)
    v = network.add_activation(u, trt.ActivationType.RELU).get_output(0)
    out = network.add_reduce(v, trt.ReduceOperation.SUM, 1 << 1, False).get_output(0)
    out.name = "out"
    network.mark_output(out)
    blob = builder.build_serialized_network(network, builder.create_builder_config())
    assert blob is not None, "engine build failed"
    return trt.Runtime(logger).deserialize_cuda_engine(blob)


def _run_larger(engine, host):
    context = engine.create_execution_context()
    tensors = [torch.from_numpy(h).cuda() for h in host]
    out = torch.empty(R, dtype=torch.float32, device="cuda")
    for name, tensor in zip(("x", "s", "r"), tensors):
        context.set_tensor_address(name, tensor.data_ptr())
    context.set_tensor_address("out", out.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out.cpu().numpy()


@pytest.mark.parametrize("kind", ["cuda", "cutile"])
def test_pattern_replaced_inside_larger_graph(kind):
    """Only the matched ``mul + add`` is replaced; the other ops stay as they are."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    kernel = _make_kernel(kind)
    rng = np.random.default_rng(0)
    weights = (rng.standard_normal((C, C)) / 8).astype(np.float32)
    host = [rng.standard_normal((R, C)).astype(np.float32) for _ in range(3)]
    p = host[0].astype(np.float64) @ weights.astype(np.float64)
    original = np.maximum(p * host[1] + host[2], 0).sum(1)
    expected = np.maximum(p * host[1] + host[2] + 1.0, 0).sum(1)

    baseline = _run_larger(_build_larger_engine(weights), host)
    np.testing.assert_allclose(baseline, original, rtol=1e-3, atol=1e-3)

    with user_kernels([kernel]):
        replaced = _run_larger(_build_larger_engine(weights), host)

    if np.allclose(replaced, baseline, rtol=1e-3, atol=1e-3):
        pytest.skip("this TensorRT build ignores the user kernel option")
    np.testing.assert_allclose(replaced, expected, rtol=1e-3, atol=1e-3)


def test_cutile_kernel_replaces_pattern():
    """Same pattern as the first test, with a cuTile kernel built into a cubin by the helper."""
    kernel = _make_kernel("cutile")
    rng = np.random.default_rng(0)
    host = [rng.standard_normal(N).astype(np.float32) for _ in range(3)]
    inputs = [torch.from_numpy(x).cuda() for x in host]
    original = host[0] * host[1] + host[2]

    baseline = _run(_build_engine(), inputs)
    with user_kernels([kernel]):
        replaced = _run(_build_engine(), inputs)

    if np.allclose(replaced, baseline, rtol=1e-5, atol=1e-6):
        pytest.skip("this TensorRT build ignores the user kernel option")
    np.testing.assert_allclose(replaced, original + 1.0, rtol=1e-5, atol=1e-6)
