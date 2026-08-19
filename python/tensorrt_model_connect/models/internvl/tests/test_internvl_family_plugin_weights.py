# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations



from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestInternVLPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 2, 32

    def _make_text_tensors(self):
        """Create synthetic text decoder weights with InternVL3 key naming."""
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        t = {}
        t["language_model.model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"language_model.model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            t[f"{p}.self_attn.q_proj.bias"] = _rand(self.HIDDEN)
            t[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            t[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP)
        t["language_model.model.norm.weight"] = _rand(self.HIDDEN)
        t["language_model.lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_load_text_weights_keys(self, tmp_path):
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "patch_size": 14,
            },
        }
        tensors = self._make_text_tensors()
        # Add vision weights that should be ignored
        tensors["visual.patch_embed.proj.weight"] = _rand(64, 3, 14, 14)
        tensors["mlp1.0.weight"] = _rand(self.HIDDEN, 64)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm", "w_q", "w_k", "w_v",
                        "w_o", "w_gate", "w_up", "w_down"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
        assert "final_norm" in weights
        assert "w_out" in weights
        assert weights["_attention_size"] == self.HIDDEN
        assert weights["_mlp_size"] == self.MLP

    def test_qkv_biases_loaded(self, tmp_path):
        """InternVL3 (Qwen2) has q/k biases."""
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            assert f"layer.{i}.q_bias" in weights
            assert f"layer.{i}.k_bias" in weights
            kv_dim = self.KV_HEADS * (self.HIDDEN // self.HEADS)
            assert weights[f"layer.{i}.k_bias"].shape == (kv_dim,)

    def test_vision_weights_not_in_text(self, tmp_path):
        """Vision and projector keys should NOT appear in text weight dict."""
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        tensors["visual.patch_embed.proj.weight"] = _rand(64, 3, 14, 14)
        tensors["mlp1.0.weight"] = _rand(self.HIDDEN, 64)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for key in weights:
            assert not key.startswith("visual."), f"Vision key leaked: {key}"
            assert not key.startswith("mlp1."), f"Projector key leaked: {key}"

    def test_transpose_applied(self, tmp_path):
        """Projections should be transposed from [out, in] to [in, out]."""
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": 1,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {},
        }
        tensors = self._make_text_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # w_q: [hidden, hidden] transposed
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, self.HIDDEN)
        # w_gate: [mlp, hidden] transposed to [hidden, mlp]
        assert weights["layer.0.w_gate"].shape == (self.HIDDEN, self.MLP)
        # w_out: [vocab, hidden] transposed to [hidden, vocab]
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_get_vl_config(self, tmp_path):
        """get_vl_config should return correct VL config for InternVL."""
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "text_config": {
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
            },
            "vision_config": {
                "hidden_size": 64,
                "patch_size": 14,
            },
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)

        assert vl_cfg is not None
        assert vl_cfg["preprocessor_type"] == "simple_chw"
        assert vl_cfg["interpolation"] == "bicubic"
        assert vl_cfg["fixed_image_size"] == 448
        # num_patches = (448/14)^2 = 1024
        assert vl_cfg["num_image_pad_tokens"] == 256
        assert vl_cfg["vision_output_dim"] == self.HIDDEN
        assert "image_token_id" in vl_cfg
        assert "vl_prompt_template" in vl_cfg

    def test_no_vl_config_without_vision(self, tmp_path):
        """get_vl_config returns None when no vision_config present."""
        import tensorrt_model_connect.models.internvl.model as plugin

        config = {
            "model_type": "internvl_chat",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)
        assert vl_cfg is None
