# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Toy example: the pattern ``a * b + c`` is replaced by a user kernel at engine build.

The graph is exactly the pattern. The user kernel computes ``a * b + c + 1``, so its output cannot
be mistaken for the output of the original ops. A TensorRT build that ignores the build option
returns the original result, and the test skips.
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
