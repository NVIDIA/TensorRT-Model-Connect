# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint layout and unsupported build-contract regression tests."""

import json
import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorrt_model_connect import BuildRequest
from families.qwen3_vl_moe.config import ModelConfig, get_rope_scaling
from families.qwen3_vl_moe.model import build, load_weights, _tokenizer_runtime_contract
from families.qwen3_vl_moe.checkpoint_mapper import _target_np_dtype


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
        "lm_head.weight": np.arange(28, dtype=np.float32).reshape(7, 4) / 100,
        prefix + ".input_layernorm.weight": np.ones(4, np.float32),
        prefix + ".post_attention_layernorm.weight": np.ones(4, np.float32),
        prefix + ".self_attn.q_norm.weight": np.ones(2, np.float32),
        prefix + ".self_attn.k_norm.weight": np.ones(2, np.float32),
        prefix + ".mlp.gate.weight": np.arange(12, dtype=np.float32).reshape(3, 4) / 100,
    }
    for name, rows in [("q_proj", 4), ("k_proj", 2), ("v_proj", 2), ("o_proj", 4)]:
        values[prefix + ".self_attn." + name + ".weight"] = (
            np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 100
        )
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
    np.testing.assert_array_equal(
        weights["layer.0.router"], values[prefix + ".mlp.gate.weight"].T.astype(dtype)
    )
    for name in ("q", "k", "v", "o"):
        np.testing.assert_array_equal(
            weights[f"layer.0.w_{name}"],
            values[f"{prefix}.self_attn.{name}_proj.weight"].T.astype(dtype),
        )
    np.testing.assert_array_equal(weights["w_out"], values["lm_head.weight"].T.astype(dtype))


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


@pytest.mark.parametrize("layout", ["rope_scaling", "rope_parameters", "missing"])
def test_multimodal_rope_is_required_before_weight_loading(tmp_path, monkeypatch, layout):
    rope = {"mrope_section": [2, 1, 1], "mrope_interleaved": True}
    text = {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }
    if layout != "missing":
        text[layout] = rope
    config = {
        "model_type": "qwen3_vl_moe",
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "text_config": text,
        "vision_config": {"deepstack_visual_indexes": [0]},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    calls = []

    def reached_weights(*args):
        calls.append(True)
        raise RuntimeError("weight loading reached")

    monkeypatch.setattr("families.qwen3_vl_moe.model.load_weights", reached_weights)
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="qwen3_vl_moe",
        task="vision_language_generation",
        precision="bf16",
        max_sequence_length=16,
    )
    if layout == "missing":
        with pytest.raises(ValueError):
            build(request, None)
        assert not calls
    else:
        assert get_rope_scaling(config) == rope
        with pytest.raises(RuntimeError, match="weight loading reached"):
            build(request, None)
        assert calls == [True]


def test_tokenizer_contract_disables_checkpoint_code(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [42]

    def load(path, *, trust_remote_code, use_fast):
        assert path == str(tmp_path) and not trust_remote_code and use_fast
        return Tokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    assert _tokenizer_runtime_contract(tmp_path) == {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": [],
        "tokenizer_suffix_ids": [],
    }


def test_standard_mapper_retains_bf16_range():
    values = np.array([2**-25, 65536], dtype=np.float32)
    stored = values.astype(_target_np_dtype("bf16"))
    assert stored.dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(stored.astype(np.float32), values)
