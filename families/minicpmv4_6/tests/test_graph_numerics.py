# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder graph helpers must agree with the reference formulas, including at their extremes."""

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.minicpmv4_6 import graph_ops  # noqa: E402
from families.minicpmv4_6.text_model import _normalized_heads  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


def _run(build, value, input_dtype=None):
    """Build an engine around ``build(network, x)`` and run it on ``value``; the output is FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    trt_dtype = trt.float16 if input_dtype == torch.float16 else trt.float32
    output = build(network, network.add_input("x", trt_dtype, tuple(value.shape)))
    assert output.dtype == trt.float32, output.dtype
    output.name = "y"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    source = value.to(input_dtype or torch.float32).contiguous().cuda()
    tensors = {"x": source, "y": torch.empty(tuple(value.shape), device="cuda")}
    for name, tensor in tensors.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return tensors["y"].cpu()


def test_the_qk_l2_norm_adds_eps_under_the_root_as_the_reference():
    # The reference's l2norm is x * rsqrt(sum(x^2) + eps); dividing by max(norm, eps)
    # instead turns these short vectors into unit vectors.
    torch.manual_seed(2046)
    value = torch.randn(4, 128) * torch.tensor([[1.0], [1e-2], [1e-3], [1e-4]])
    actual = _run(lambda network, x: graph_ops.add_l2_norm(network, x, reduce_axis=1, eps=1e-6), value)
    expected = value * torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_softplus_stays_finite_and_matches_torch():
    # log(1 + exp(x)) overflows FP32 above x = 88; the decay then turns into NaN.
    value = torch.tensor([[-100.0, -30.0, -5.0, -1e-3, 0.0, 1e-3, 5.0, 20.0, 30.0, 88.0, 90.0, 100.0, 1e4]])
    actual = _run(graph_ops.add_softplus, value)
    assert torch.isfinite(actual).all(), actual
    torch.testing.assert_close(actual, torch.nn.functional.softplus(value), atol=1e-6, rtol=1e-6)


def test_the_head_rms_norm_uses_the_given_eps():
    torch.manual_seed(7)
    value = torch.randn(3, 4, 8) * 0.1
    actual = _run(lambda network, x: graph_ops.add_rms_norm_last_axis(network, x, 0.5), value)
    expected = value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 0.5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_normalized_fp16_heads_stay_fp32_for_the_recurrence():
    # The reference feeds its l2norm output to the FP32 recurrence without rounding it to FP16;
    # an FP16 round trip would leave errors of about 1e-3 here.
    torch.manual_seed(11)
    value = torch.randn(2, 4, 128).half()
    actual = _run(lambda network, x: _normalized_heads(network, x, 2), value, input_dtype=torch.float16)
    reference = value.float()
    expected = reference * torch.rsqrt((reference * reference).sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
