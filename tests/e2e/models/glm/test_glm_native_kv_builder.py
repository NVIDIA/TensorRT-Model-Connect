# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-owned TensorRT build proof for split native-KV engines."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.glm.config import ModelConfig  # noqa: E402

plugin_module = importlib.import_module("tensorrt_model_connect.families.glm.plugin")


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="glm",
        architectures=["GlmForCausalLM"],
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1.5625e-7,
        rope_theta=10000.0,
        max_position_embeddings=256,
        hidden_act="silu",
        _head_dim=128,
        raw={
            "attention_bias": True,
            "partial_rotary_factor": 0.5,
            "_decoder_engine_layout": "split",
        },
    )


def _weights(config: ModelConfig) -> dict[str, object]:
    rng = np.random.default_rng(20260728)
    hidden = config.hidden_size
    attention = config.num_attention_heads * 128
    kv_attention = config.num_key_value_heads * 128
    mlp = config.intermediate_size

    def weight(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float16) * 0.01

    return {
        "embedding": weight(config.vocab_size, hidden),
        "layer.0.input_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.w_q": weight(hidden, attention),
        "layer.0.q_bias": weight(attention),
        "layer.0.w_k": weight(hidden, kv_attention),
        "layer.0.k_bias": weight(kv_attention),
        "layer.0.w_v": weight(hidden, kv_attention),
        "layer.0.v_bias": weight(kv_attention),
        "layer.0.w_o": weight(attention, hidden),
        "layer.0.post_attn_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.w_gate": weight(hidden, mlp),
        "layer.0.w_up": weight(hidden, mlp),
        "layer.0.w_down": weight(mlp, hidden),
        "final_norm": np.ones(hidden, dtype=np.float32),
        "w_out": weight(hidden, config.vocab_size),
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_mlp_size": mlp,
    }


@pytest.mark.parametrize(
    ("role", "token_max"),
    [("prefill", 16), ("decode", 1)],
)
@requires_trt
def test_split_native_engine_builds_with_aliased_full_capacity_cache(
    role: str,
    token_max: int,
) -> None:
    config = _config()
    plan = plugin_module.build_native_decoder_engine(
        config,
        _weights(config),
        256,
        opt_prefill_length=4,
        max_prefill_length=16,
        profile_mode=role,
    )

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)

    assert engine is not None
    assert engine.get_tensor_shape("cache_k_0") == (1, 1, 256, 128)
    assert engine.get_tensor_shape("present_k_0") == (1, 1, 256, 128)
    assert engine.get_aliased_input_tensor("present_k_0") == "cache_k_0"
    assert engine.get_aliased_input_tensor("present_v_0") == "cache_v_0"
    assert engine.get_tensor_dtype("cache_write_indices") == trt.int32
    assert engine.get_tensor_dtype("key_value_lengths") == trt.int32
    assert engine.get_tensor_shape("cache_write_indices") == (1,)
    assert engine.get_tensor_shape("key_value_lengths") == (1,)
    tensor_names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    assert "attention_mask" not in tensor_names
    assert engine.get_tensor_profile_shape(
        "token_id",
        0,
    )[2] == (token_max,)
