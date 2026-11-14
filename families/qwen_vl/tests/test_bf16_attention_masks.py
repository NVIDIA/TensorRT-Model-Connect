# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-cache BF16 decoder attention accepts the family's FP32 causal masks."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.qwen_vl import graph_ops  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.trt,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")]


@pytest.fixture(scope="module", params=[16, 32])
def decoder_attention(request):
    heads, kv_heads, dimension = request.param, 8, 128
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    q = network.add_input("q", trt.bfloat16, (-1, heads * dimension))
    k = network.add_input("k", trt.bfloat16, (-1, kv_heads * dimension))
    v = network.add_input("v", trt.bfloat16, (-1, kv_heads * dimension))
    mask = network.add_input("mask", trt.float32, (-1, -1))
    output = graph_ops.add_attention_from_rows(
        network, q, k, v, num_heads=heads, num_kv_heads=kv_heads, head_dim=dimension,
        q_seq=None, kv_seq=None, mask=graph_ops.add_2d_mask_to_4d(network, mask))
    output.name = "output"
    network.mark_output(output)
    profile = builder.create_optimization_profile()
    profile.set_shape("q", (1, heads * dimension), (64, heads * dimension), (128, heads * dimension))
    for name in ("k", "v"):
        profile.set_shape(name, (512, kv_heads * dimension), (1024, kv_heads * dimension),
                          (33280, kv_heads * dimension))
    profile.set_shape("mask", (1, 512), (64, 1024), (128, 33280))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "BF16 attention must accept the decoder's FP32 causal mask"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    yield engine, heads, kv_heads, dimension


@pytest.mark.parametrize("keys", [512, 513, 16641])
def test_masked_stale_values_do_not_enter_bf16_decoder_context(decoder_attention, keys):
    engine, heads, kv_heads, dimension = decoder_attention
    valid = 300
    rng = np.random.default_rng(keys + heads)
    values = {"q": rng.standard_normal((1, heads * dimension)).astype(np.float32),
              "k": rng.standard_normal((keys, kv_heads * dimension)).astype(np.float32),
              "v": rng.standard_normal((keys, kv_heads * dimension)).astype(np.float32)}
    values["k"][valid:] = 30.0
    values["v"][valid:] = 100.0
    buffers = {name: torch.from_numpy(array).to(device="cuda", dtype=torch.bfloat16)
               for name, array in values.items()}
    rounded = {name: tensor.float().cpu().numpy().astype(np.float64)
               for name, tensor in buffers.items()}
    mask = np.zeros((1, keys), dtype=np.float32)
    mask[:, valid:] = -1e9
    buffers["mask"] = torch.from_numpy(mask).cuda()
    context = engine.create_execution_context()
    assert context is not None
    for name, tensor in buffers.items():
        assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty((1, heads * dimension), device="cuda", dtype=torch.bfloat16)
    assert context.set_tensor_address("output", output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    query = rounded["q"].reshape(heads, dimension)
    key = rounded["k"][:valid].reshape(valid, kv_heads, dimension)
    value = rounded["v"][:valid].reshape(valid, kv_heads, dimension)
    expected = []
    for head in range(heads):
        kv_head = head // (heads // kv_heads)
        score = key[:, kv_head] @ query[head] / np.sqrt(dimension)
        probability = np.exp(score - score.max())
        probability /= probability.sum()
        expected.append(probability @ value[:, kv_head])
    # Start from the actual BF16 inputs and allow BF16 query/output rounding.
    # Masked stale values would cause errors near 100.
    np.testing.assert_allclose(output.float().cpu().numpy(), np.asarray(expected).reshape(1, -1),
                               rtol=2e-2, atol=3e-3)
