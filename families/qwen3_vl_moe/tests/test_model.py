# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint layout and unsupported build-contract regression tests."""

import json
import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorrt_model_connect import BuildRequest
from families.qwen3_vl_moe.config import ModelConfig
from families.qwen3_vl_moe.model import build, load_weights


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_packed_checkpoint_uses_input_output_layout(tmp_path, precision):
    text = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 7,
        "num_experts": 3,
        "moe_intermediate_size": 6,
        "rms_norm_eps": 1e-6,
    }
    cfg = ModelConfig.from_json(json.dumps({"model_type": "qwen3_vl_moe", "text_config": text}))
    prefix = "model.language_model.layers.0"
    values = {
        "model.language_model.embed_tokens.weight": np.zeros((7, 4), np.float32),
        "model.language_model.norm.weight": np.ones(4, np.float32),
        "lm_head.weight": np.zeros((7, 4), np.float32),
        prefix + ".input_layernorm.weight": np.ones(4, np.float32),
        prefix + ".post_attention_layernorm.weight": np.ones(4, np.float32),
        prefix + ".self_attn.q_norm.weight": np.ones(2, np.float32),
        prefix + ".self_attn.k_norm.weight": np.ones(2, np.float32),
        prefix + ".mlp.gate.weight": np.zeros((3, 4), np.float32),
    }
    for name, rows in [("q_proj", 4), ("k_proj", 2), ("v_proj", 2), ("o_proj", 4)]:
        values[prefix + ".self_attn." + name + ".weight"] = np.zeros((rows, 4), np.float32)
    packed = np.arange(3 * 4 * 12, dtype=np.float32).reshape(3, 4, 12) / 100
    packed[0, 0, 0] = 2**-25
    down = np.arange(3 * 6 * 4, dtype=np.float32).reshape(3, 6, 4) / 100
    values[prefix + ".mlp.experts.gate_up_proj"] = packed
    values[prefix + ".mlp.experts.down_proj"] = down
    save_file(values, tmp_path / "model.safetensors")
    weights = load_weights(tmp_path, cfg, precision)
    dtype = ml_dtypes.bfloat16 if precision == "bf16" else np.float32
    np.testing.assert_array_equal(weights["layer.0.experts.w_gate"], packed[:, :, :6].astype(dtype))
    np.testing.assert_array_equal(weights["layer.0.experts.w_up"], packed[:, :, 6:].astype(dtype))
    np.testing.assert_array_equal(weights["layer.0.experts.w_down"], down.astype(dtype))


@pytest.mark.parametrize(
    "override,error",
    [
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"image_height": 448}, NotImplementedError),
        ({"precision": "fp16"}, ValueError),
        ({"task": "text_generation"}, ValueError),
    ],
)
def test_unsupported_build_options_fail_before_loading(tmp_path, override, error):
    args = dict(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="qwen3_vl_moe",
        task="vision_language_generation",
        precision="bf16",
    )
    args.update(override)
    with pytest.raises(error):
        build(BuildRequest(**args), None)
