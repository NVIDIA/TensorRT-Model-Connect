# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single-token decoder must respect the whole masked GQA cache."""

import math

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.starcoder2 import graph_ops  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


@pytest.mark.parametrize(
    ("cache_length", "previous_tokens", "heads", "kv_heads"),
    [
        (256, 0, 24, 2),
        (512, 0, 24, 2),
        (512, 7, 24, 2),
        (512, 255, 24, 2),
        (512, 0, 36, 4),
        (512, 7, 36, 4),
        (2048, 1023, 24, 2),
    ],
)
def test_fp16_decode_attention_matches_masked_reference(
    cache_length, previous_tokens, heads, kv_heads
):
    head_dim = 128
    key_length = cache_length + 1
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    q = network.add_input("q", trt.float16, (1, heads * head_dim))
    k = network.add_input("k", trt.float16, (key_length, kv_heads * head_dim))
    v = network.add_input("v", trt.float16, (key_length, kv_heads * head_dim))
    mask_input = network.add_input("mask", trt.float32, (1, key_length))
    mask_half = network.add_cast(mask_input, trt.float16).get_output(0)
    mask = graph_ops.add_2d_mask_to_4d(network, mask_half)
    out = graph_ops.add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=key_length,
        causal=False,
        mask=mask,
        scale=1.0 / math.sqrt(head_dim),
    )
    out.name = "output"
    network.mark_output(out)
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "FP16 decode attention must build"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None

    generator = torch.Generator(device="cpu").manual_seed(2049 + previous_tokens)
    buffers = {
        "q": torch.randn((1, heads * head_dim), generator=generator, dtype=torch.float16).cuda(),
        "k": torch.randn(
            (key_length, kv_heads * head_dim), generator=generator, dtype=torch.float16
        ).cuda(),
        "v": torch.randn(
            (key_length, kv_heads * head_dim), generator=generator, dtype=torch.float16
        ).cuda(),
        "mask": torch.full((1, key_length), -10000.0, dtype=torch.float32, device="cuda"),
        "output": torch.empty((1, heads * head_dim), dtype=torch.float16, device="cuda"),
    }
    buffers["mask"][..., :previous_tokens] = 0
    buffers["mask"][..., -1] = 0
    for name, tensor in buffers.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    query = buffers["q"].view(1, heads, 1, head_dim).float()
    key = buffers["k"].view(key_length, kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
    value = buffers["v"].view(key_length, kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
    key = key.repeat_interleave(heads // kv_heads, dim=1).float()
    value = value.repeat_interleave(heads // kv_heads, dim=1).float()
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(head_dim) + buffers["mask"][
        :, None, None, :
    ]
    expected = (scores.softmax(-1) @ value).reshape(1, heads * head_dim)
    torch.testing.assert_close(buffers["output"].float(), expected, rtol=2e-3, atol=2e-3)
