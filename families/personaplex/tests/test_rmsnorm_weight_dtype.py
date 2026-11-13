# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the RMSNorm weight dtype through the norm's normal output."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.personaplex import graph_ops  # noqa: E402

# transformer.layers.0.norm1.alpha[65] of the PersonaPlex checkpoint: exact in
# BF16, but a Half subnormal rounds it to -1.0132789611816406e-06.
COEFFICIENT = -1.0207295417785645e-06
EPS = 1e-8
# x = [3, 1, 1, 1] gives mean(x * x) = 3 exactly at every width, so neither
# the reduction order nor the operation grouping can change the result.
X_BLOCK = (3.0, 1.0, 1.0, 1.0)
# The pinned Moshi _rms_norm on CUDA with the weight stored in Half (the
# reference loads the model in FP16): sqrt(3) * Half(coefficient) and 1/sqrt(3),
# each rounded to Half. An FP32 weight gives -1.7881393432617188e-06 instead.
EXPECTED_HALF_BLOCK = (-1.7285346984863281e-06, 0.5771484375, 0.5771484375, 0.5771484375)


def _run(width, dtype):
    trt_dtype = trt.float16 if dtype == np.float16 else trt.float32
    torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt_dtype, (1, width))
    eps = graph_ops.add_constant(network, (1, 1), np.array([[EPS]], np.float32), dtype=np.float32)
    gamma = np.tile(np.array([COEFFICIENT, 1.0, 1.0, 1.0], np.float32), width // 4)
    y = graph_ops.add_rms_norm(network, x, width, gamma, eps, dtype=dtype)
    y.name = "y"
    network.mark_output(y)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    x_value = (
        torch.tensor(X_BLOCK, dtype=torch_dtype, device="cuda").repeat(width // 4).reshape(1, width)
    )
    y_value = torch.empty((1, width), dtype=torch_dtype, device="cuda")
    assert context.set_tensor_address("x", x_value.data_ptr())
    assert context.set_tensor_address("y", y_value.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = y_value.cpu().reshape(-1)
    del context, engine, runtime
    return actual


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("rebuild", range(3))
@pytest.mark.parametrize("width", [4, 4096])
def test_half_rmsnorm_uses_the_half_weight(width, rebuild):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    del rebuild  # Each parametrized invocation builds a separate engine.
    actual = _run(width, np.float16)
    expected = torch.tensor(EXPECTED_HALF_BLOCK, dtype=torch.float16).repeat(width // 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp32_rmsnorm_keeps_the_fp32_weight():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    actual = _run(4, np.float32)
    # Half storage would move this value by 0.73%; FP32 rounding noise is ~1e-7.
    expected = np.sqrt(3.0) * COEFFICIENT
    assert abs(float(actual[0]) - expected) <= 1e-5 * abs(expected)
