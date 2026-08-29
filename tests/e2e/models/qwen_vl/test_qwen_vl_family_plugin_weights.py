# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


def _make_standard_decoder_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim
    tensors = {}
    tensors["model.embed_tokens.weight"] = _rand(vocab, hidden)
    for i in range(layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = _rand(hidden)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(hidden)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
        tensors[f"{prefix}.mlp.up_proj.weight"] = _rand(mlp, hidden)
        tensors[f"{prefix}.mlp.down_proj.weight"] = _rand(hidden, mlp)
    tensors["model.norm.weight"] = _rand(hidden)
    tensors["lm_head.weight"] = _rand(vocab, hidden)
    return tensors


class TestQwenVLPlugin:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    def test_qwen25_vl_delegates_to_standard(self, tmp_path):
        """Qwen2.5-VL (no deepstack) delegates to load_standard_weights."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen2_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
        }
        tensors = _make_standard_decoder_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS,
            self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        for i in range(self.LAYERS):
            assert f"layer.{i}.w_q" in weights
        assert "final_norm" in weights
        assert "w_out" in weights
        assert plugin.get_bundle_config_overrides(cfg) is None

    def test_qwen3_vl_language_model_prefix(self, tmp_path):
        """Qwen3-VL uses model.language_model.layers.{i} prefix."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen3_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0, 1],
            },
        }
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        tensors = {}
        tensors["model.language_model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"model.language_model.layers.{i}"
            tensors[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
            tensors[f"{p}.self_attn.q_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.self_attn.k_proj.weight"] = _rand(
                kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.v_proj.weight"] = _rand(
                kv_hidden, self.HIDDEN)
            tensors[f"{p}.self_attn.o_proj.weight"] = _rand(
                self.HIDDEN, self.HIDDEN)
            tensors[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
            tensors[f"{p}.mlp.down_proj.weight"] = _rand(
                self.HIDDEN, self.MLP)
        tensors["model.language_model.norm.weight"] = _rand(self.HIDDEN)
        tensors["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)

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

    def test_qwen3_vl_vision_weights_not_in_text(self, tmp_path):
        """Vision weights (visual.*) should NOT appear in text weight dict."""
        from tensorrt_model_connect.families.qwen_vl import plugin

        config = {
            "model_type": "qwen3_vl",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0],
            },
        }
        head_dim = self.HIDDEN // self.HEADS
        kv_hidden = self.KV_HEADS * head_dim
        tensors = {}
        tensors["model.language_model.embed_tokens.weight"] = _rand(
            self.VOCAB, self.HIDDEN)
        p = "model.language_model.layers.0"
        tensors[f"{p}.input_layernorm.weight"] = _rand(self.HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = _rand(self.HIDDEN)
        tensors[f"{p}.self_attn.q_proj.weight"] = _rand(
            self.HIDDEN, self.HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = _rand(
            kv_hidden, self.HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = _rand(
            kv_hidden, self.HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = _rand(
            self.HIDDEN, self.HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = _rand(self.MLP, self.HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = _rand(self.MLP, self.HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = _rand(self.HIDDEN, self.MLP)
        tensors["model.language_model.norm.weight"] = _rand(self.HIDDEN)
        tensors["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        # Add some vision weights that should be ignored by text loader
        tensors["model.visual.patch_embed.weight"] = _rand(16, 3, 14, 14)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Vision keys should not appear in text weights
        for key in weights:
            assert not key.startswith("visual."), f"Vision key leaked: {key}"
            assert not key.startswith("model.visual."), \
                f"Vision key leaked: {key}"

    def test_qwen3_vl_mock_bundle_serializes_decoder_geometry_at_top_level(
        self,
        tmp_path,
    ):
        """Exercise the pinned Qwen3-VL producer contract with mocked engines."""
        from tensorrt_model_connect.engine_builder import build_bundle
        from tensorrt_model_connect.families.qwen_vl.plugin import (
            plugin as production_plugin,
        )

        # Qwen/Qwen3-VL-2B-Instruct at
        # 89644892e4d85e24eaac8bacfd4f463576704203.
        source_config = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "image_token_id": 151655,
            "model_type": "qwen3_vl",
            "text_config": {
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000,
                "vocab_size": 151936,
            },
            "vision_config": {
                "deepstack_visual_indexes": [5, 11, 17],
                "hidden_size": 1024,
                "out_hidden_size": 2048,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
        _write_config(tmp_path, source_config)
        (tmp_path / "generation_config.json").write_text(
            json.dumps(
                {
                    "bos_token_id": 151643,
                    "eos_token_id": [151645, 151643],
                    "pad_token_id": 151643,
                }
            ),
            encoding="utf-8",
        )

        class MockQwenVLPlugin:
            name = "qwen_vl"
            runtime_strategy = "qwen_vl_vision_language"
            embed_input = True
            requires_tokenizer = False

            @staticmethod
            def load_weights(_model_dir, _config):
                return {}

            @staticmethod
            def build_engine(_config, _weights, _max_cache_length, **_kwargs):
                return b"MOCK_DECODER_PLAN"

            @staticmethod
            def build_vision_engine(_model_dir, _config, _weights, **_kwargs):
                return b"MOCK_VISION_PLAN"

            @staticmethod
            def get_vl_config(config):
                return production_plugin.get_vl_config(config)

            @staticmethod
            def get_bundle_config_overrides(config):
                return production_plugin.get_bundle_config_overrides(config)

        with (
            patch(
                "tensorrt_model_connect.engine_builder.find_plugin",
                return_value=MockQwenVLPlugin(),
            ),
            patch(
                "tensorrt_model_connect.engine_builder._get_trt_version",
                return_value="11.1.0",
            ),
            patch(
                "tensorrt_model_connect.engine_builder._get_gpu_name",
                return_value="CPU unit mock",
            ),
            patch("tensorrt_model_connect.engine_builder.write_bundle") as write_bundle,
        ):
            build_bundle(
                str(tmp_path),
                str(tmp_path / "qwen3-vl-2b.bundle"),
                max_cache_length=256,
            )

        sections = {
            section.name: section.data for section in write_bundle.call_args.args[2]
        }
        runtime_config = json.loads(sections["config.json"])
        decoder_config = source_config["text_config"]
        assert runtime_config["text_config"] == decoder_config
        decoder_contract = {
            key: decoder_config[key]
            for key in (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "bos_token_id",
            )
        }
        assert all(key not in source_config for key in decoder_contract)
        assert {key: runtime_config[key] for key in decoder_contract} == decoder_contract
        assert runtime_config["eos_token_id"] == [151645, 151643]
