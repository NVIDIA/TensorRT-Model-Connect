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


class TestLocateAnythingPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 64, 16, 2, 4, 2, 32

    def _make_text_tensors(self):
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        tensors = {}
        tensors["language_model.model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"language_model.model.layers.{i}"
            tensors[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.self_attn.q_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.o_proj.weight"] = _rand(self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.self_attn.q_proj.bias"] = _rand(self.HIDDEN)
            tensors[f"{p}.self_attn.k_proj.bias"] = _rand(kv_hidden)
            tensors[f"{p}.self_attn.v_proj.bias"] = _rand(kv_hidden)
            tensors[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP)
        tensors["language_model.model.norm.weight"] = _rand(self.HIDDEN)
        tensors["language_model.lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return tensors

    def _write_locateanything_config(self, tmp_path):
        config = {
            "model_type": "locateanything",
            "architectures": ["LocateAnythingForConditionalGeneration"],
            "image_token_index": 151665,
            "box_start_token_id": 151668,
            "box_end_token_id": 151669,
            "coord_start_token_id": 151677,
            "coord_end_token_id": 152677,
            "text_config": {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": self.VOCAB,
                "hidden_size": self.HIDDEN,
                "intermediate_size": self.MLP,
                "num_hidden_layers": self.LAYERS,
                "num_attention_heads": self.HEADS,
                "num_key_value_heads": self.KV_HEADS,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 32768,
            },
            "vision_config": {
                "model_type": "moonvit",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
            },
        }
        _write_config(tmp_path, config)

    def test_load_text_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.locateanything import plugin

        self._write_locateanything_config(tmp_path)
        tensors = self._make_text_tensors()
        tensors["vision_model.patch_embed.proj.weight"] = _rand(64, 3, 14, 14)
        tensors["mlp1.1.weight"] = _rand(self.HIDDEN, 64 * 4)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert cfg.model_type == "locateanything"
        assert cfg.hidden_size == self.HIDDEN
        assert "embedding" in weights
        for i in range(self.LAYERS):
            for key in ("input_norm", "post_attn_norm", "w_q", "w_k", "w_v",
                        "w_o", "w_gate", "w_up", "w_down"):
                assert f"layer.{i}.{key}" in weights
            assert f"layer.{i}.q_bias" in weights
            assert f"layer.{i}.k_bias" in weights
            assert f"layer.{i}.v_bias" in weights
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)
        assert weights["_kv_attention_size"] == self.KV_HEADS * (
            self.HIDDEN // self.HEADS)

    def test_vision_weights_not_in_text(self, tmp_path):
        from tensorrt_model_connect.families.locateanything import plugin

        self._write_locateanything_config(tmp_path)
        tensors = self._make_text_tensors()
        tensors["vision_model.encoder.blocks.0.wqkv.weight"] = _rand(192, 64)
        tensors["vision_model.patch_embed.pos_emb.weight"] = _rand(64, 64, 64)
        tensors["mlp1.1.weight"] = _rand(self.HIDDEN, 64 * 4)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for key in weights:
            assert not key.startswith("vision_model."), f"Vision key leaked: {key}"
            assert not key.startswith("mlp1."), f"Projector key leaked: {key}"

    def test_get_vl_config(self, tmp_path):
        from tensorrt_model_connect.families.locateanything import plugin

        self._write_locateanything_config(tmp_path)
        cfg = ModelConfig.from_dir(tmp_path)
        vl_cfg = plugin.get_vl_config(cfg)

        assert vl_cfg is not None
        assert vl_cfg["image_token_id"] == 151665
        assert vl_cfg["fixed_image_size"] == 448
        assert vl_cfg["num_image_pad_tokens"] == 256
        assert vl_cfg["vision_output_dim"] == self.HIDDEN
        assert vl_cfg["preprocessor_type"] == "patchify_chw"
        assert vl_cfg["patch_size"] == 14
        assert vl_cfg["merge_size"] == 2
        assert vl_cfg["image_mean"] == [0.5, 0.5, 0.5]
        assert vl_cfg["image_std"] == [0.5, 0.5, 0.5]
        assert vl_cfg["image_token_str"] == "<IMG_CONTEXT>"
        assert "locateanything_vision_engine_supported" not in vl_cfg
        assert vl_cfg["box_start_token_id"] == 151668
        assert "{image_pads}" in vl_cfg["vl_prompt_template"]
        assert "{prompt}" in vl_cfg["vl_prompt_template"]
