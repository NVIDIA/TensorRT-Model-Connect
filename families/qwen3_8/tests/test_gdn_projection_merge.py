# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One FP8 GEMM for two projections (rows concatenated, one weight scale per row) vs two GEMMs.

Needs a CUDA GPU with FP8 support and TensorRT; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip("CUDA GPU with FP8 support required", allow_module_level=True)

from families.qwen3_8 import engine_builder as EB  # noqa: E402
from families.qwen3_8 import quantization as Q  # noqa: E402

K, N1, N2 = 512, 192, 96


def fp8_layer(n, weight_scale, input_scale, seed):
    gen = torch.Generator().manual_seed(seed)
    packed = (torch.randn(n, K, generator=gen) * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
    return Q._LayerScales(
        input_scale=input_scale,
        weight=Q._FP8Weight(packed=packed.numpy(), weight_scale=weight_scale, out_features=n,
                            in_features=K))


def run(ctx, x, build):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inp = net.add_input("x", trt.bfloat16, (1, K))
    outs = build(net, inp)
    for i, out in enumerate(outs):
        out.name = f"out{i}"
        net.mark_output(out)
    blob = builder.build_serialized_network(net, builder.create_builder_config())
    assert blob is not None
    exec_ctx = trt.Runtime(logger).deserialize_cuda_engine(blob).create_execution_context()
    results = [torch.empty(tuple(out.shape), device="cuda", dtype=torch.bfloat16) for out in outs]
    exec_ctx.set_tensor_address("x", x.data_ptr())
    for i, r in enumerate(results):
        exec_ctx.set_tensor_address(f"out{i}", r.data_ptr())
    assert exec_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return results


def make_ctx(scales):
    return Q.Qwen38QuantContext(profile=Q._Profile(scales), graph_ops=EB.graph_ops)


def test_merged_fp8_matmul_equals_separate():
    scales = {"p.qkv": fp8_layer(N1, 0.02, 0.05, 1), "p.z": fp8_layer(N2, 0.05, 0.05, 2)}
    ctx = make_ctx(scales)
    x = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    weights = np.zeros((K, 1), dtype=np.float32)  # unused for quantized layers

    def separate(net, inp):
        return [ctx.maybe_quantized_matmul(net, inp, K, n, weights, name, dtype=np.float16)
                for name, n in (("p.qkv", N1), ("p.z", N2))]

    def merged(net, inp):
        out = ctx.merged_fp8_matmul(net, inp, ["p.qkv", "p.z"], dtype=np.float16)
        assert out is not None
        return [out]

    a, b = run(ctx, x, separate)
    (m,) = run(ctx, x, merged)
    assert m.shape == (1, N1 + N2)
    # the weight scales differ 2.5x, so a wrong (shared) scale would be far off
    assert (m[:, :N1].float() - a.float()).abs().max().item() <= 2e-2 * a.float().abs().max().item()
    assert (m[:, N1:].float() - b.float()).abs().max().item() <= 2e-2 * b.float().abs().max().item()


def test_merged_fp8_matmul_needs_equal_input_scales():
    scales = {"p.qkv": fp8_layer(N1, 0.02, 0.05, 1), "p.z": fp8_layer(N2, 0.05, 0.06, 2)}
    logger = trt.Logger(trt.Logger.ERROR)
    net = trt.Builder(logger).create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inp = net.add_input("x", trt.bfloat16, (1, K))
    assert make_ctx(scales).merged_fp8_matmul(net, inp, ["p.qkv", "p.z"]) is None
