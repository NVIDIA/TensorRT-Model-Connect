# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Masked BART decoder attention must agree with its mathematical definition."""

import math

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.bart.model import _add_bart_decoder_attention  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


@pytest.mark.parametrize(
    ("kv_len", "valid", "bias"),
    [
        pytest.param(257, (256,), (0.0,), id="short-self"),
        pytest.param(513, (512,), (0.0,), id="long-self-first-token"),
        pytest.param(1024, (0, 1023), (0.0, 0.0), id="long-cross"),
        pytest.param(1025, (0, 1024), (0.0, 0.0), id="long-self-cached-token"),
        pytest.param(1025, (0, 1024), (0.25, -0.5), id="long-self-additive-bias"),
    ],
)
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_masked_decoder_attention_matches_reference(kv_len, valid, bias, precision):
    heads, head_dim = 12, 64
    hidden = heads * head_dim
    trt_dtype = trt.float16 if precision == "fp16" else trt.float32
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    q = network.add_input("q", trt_dtype, (1, hidden))
    k = network.add_input("k", trt_dtype, (kv_len, hidden))
    v = network.add_input("v", trt_dtype, (kv_len, hidden))
    mask = network.add_input("mask", trt_dtype, (1, 1, 1, kv_len))
    result = _add_bart_decoder_attention(
        network, q, k, v, num_heads=heads, head_dim=head_dim, kv_seq=kv_len, mask=mask
    )
    result.name = "output"
    network.mark_output(result)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    torch.manual_seed(2046)
    tensors = {
        "q": torch.randn((1, hidden), dtype=torch_dtype, device="cuda"),
        "k": torch.randn((kv_len, hidden), dtype=torch_dtype, device="cuda"),
        "v": torch.randn((kv_len, hidden), dtype=torch_dtype, device="cuda"),
        "mask": torch.full((1, 1, 1, kv_len), -10000, dtype=torch_dtype, device="cuda"),
        "output": torch.empty((1, hidden), dtype=torch_dtype, device="cuda"),
    }
    tensors["mask"][..., list(valid)] = torch.tensor(bias, dtype=torch_dtype, device="cuda")
    for name, tensor in tensors.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    # Compute only over valid rows in FP32; masked cache contents must not contribute.
    query = tensors["q"].float().reshape(heads, 1, head_dim)
    key = tensors["k"][list(valid)].float().reshape(len(valid), heads, head_dim).transpose(0, 1)
    value = tensors["v"][list(valid)].float().reshape(len(valid), heads, head_dim).transpose(0, 1)
    scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(head_dim)
    scores += torch.tensor(bias, dtype=torch.float32, device="cuda")
    expected = torch.matmul(torch.softmax(scores, dim=-1), value).reshape(1, hidden)
    torch.cuda.synchronize()
    tolerance = 2e-3 if precision == "fp16" else 1e-5
    torch.testing.assert_close(tensors["output"].float(), expected, atol=tolerance, rtol=tolerance)
