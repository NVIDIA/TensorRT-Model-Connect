# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32-accumulation decode attention keeps the FP16 mask FP16 where TensorRT 11.4 needs it."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.stablelm import default_decoder, graph_ops  # noqa: E402
from families.stablelm.checkpoint_mapper import WeightDict  # noqa: E402
from families.stablelm.config import ModelConfig  # noqa: E402


class _FakeConfig:
    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes


@pytest.mark.parametrize(
    ("all_build_routes", "expected"),
    [
        ("-mha_fusion:gemv_mha_v2=on|off -mha_fusion:v2=on|off", True),
        ("-mha_fusion:v2=on|off", False),
        ("", False),
    ],
)
def test_keep_fp16_mask_only_when_the_knob_is_listed(all_build_routes, expected) -> None:
    assert default_decoder._keep_fp16_attention_mask(_FakeConfig(all_build_routes)) is expected
    assert default_decoder._keep_fp16_attention_mask(object()) is False


def _mask_is_cast_to_fp32(fp32_accumulation: bool, keep_fp16_mask: bool) -> bool:
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(0)
    q = network.add_input("q", trt.float16, (1, 4, 1, 8))
    k = network.add_input("k", trt.float16, (1, 4, 8, 8))
    v = network.add_input("v", trt.float16, (1, 4, 8, 8))
    mask = network.add_input("mask", trt.float16, (1, 1, 1, 8))
    graph_ops.add_attention_core(
        network,
        q,
        k,
        v,
        mask=mask,
        fp32_accumulation=fp32_accumulation,
        fp16_mask_with_fp32_accumulation=keep_fp16_mask,
    )
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    return any(
        layer.type == trt.LayerType.CAST
        and layer.get_input(0).name == "mask"
        and layer.get_output(0).dtype == trt.float32
        for layer in layers
    )


@pytest.mark.parametrize(
    ("fp32_accumulation", "keep_fp16_mask", "expected"),
    [
        (True, False, True),  # default and TensorRT 11.1: the mask is cast to FP32
        (True, True, False),  # TensorRT 11.4: no mixed half/float add
        (False, True, False),  # FP16 attention never casts the mask
        (False, False, False),
    ],
)
def test_attention_mask_cast(fp32_accumulation, keep_fp16_mask, expected) -> None:
    assert _mask_is_cast_to_fp32(fp32_accumulation, keep_fp16_mask) is expected


class _AttentionReached(Exception):
    pass


@pytest.mark.parametrize("keep_fp16_mask", [True, False])
def test_decode_builder_passes_the_mask_policy_to_attention(monkeypatch, keep_fp16_mask) -> None:
    hidden, heads, head_dim, vocab = 32, 4, 8, 16
    config = ModelConfig(
        model_type="stablelm",
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw={"partial_rotary_factor": 0.25, "_decoder_engine_role": "decode"},
    )
    weights = WeightDict(
        embedding=np.zeros((vocab, hidden), dtype=np.float32),
        **{
            "layer.0.input_norm": np.ones(hidden, dtype=np.float32),
            "layer.0.input_norm_beta": np.zeros(hidden, dtype=np.float32),
            "layer.0.w_q": np.zeros((hidden, heads * head_dim), dtype=np.float32),
            "layer.0.w_k": np.zeros((hidden, heads * head_dim), dtype=np.float32),
            "layer.0.w_v": np.zeros((hidden, heads * head_dim), dtype=np.float32),
        },
    )
    captured: dict[str, object] = {}

    def fake_attention_core(network, q_4d, k_4d, v_4d, *, mask, **kwargs):
        # Read the mask dtype now: the network owns the tensor and is gone after the build.
        captured.update(kwargs, mask_dtype=mask.dtype)
        raise _AttentionReached

    monkeypatch.setattr(
        default_decoder, "_keep_fp16_attention_mask", lambda trt_config: keep_fp16_mask
    )
    monkeypatch.setattr(graph_ops, "add_attention_core", fake_attention_core)

    with pytest.raises(_AttentionReached):
        default_decoder.build_standard_decoder_engine(
            config,
            weights,
            max_cache_length=8,
            precision="fp16",
            norm_type="layernorm",
            partial_rotary_factor=0.25,
            fp32_attention_accumulation=True,
        )

    assert captured["fp32_accumulation"] is True
    assert captured["fp16_mask_with_fp32_accumulation"] is keep_fp16_mask
    assert captured["mask_dtype"] == trt.float16
