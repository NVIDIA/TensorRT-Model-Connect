# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base tester class for family plugin engine tests.

Provides TinyModelSpec (tiny model dimensions for testing) and FamilyPluginTester
(base class that per-family tests subclass to specify their plugin, model_type,
and family-specific weight layouts).

Usage in per-family test files:
    class ExamplePluginTester(FamilyPluginTester):
        plugin_module = "tensorrt_model_connect.models.example"
        model_type = "example_decoder"

    class TestExampleEngine(FamilyPluginTestMixin):
        tester_class = ExamplePluginTester
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
except (ImportError, ModuleNotFoundError):
    pytest.skip("safetensors not available", allow_module_level=True)

try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.checkpoint_mapper import WeightDict
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


# Fixed seed RNG for reproducible synthetic weights.
RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    """Generate a random float32 tensor with reproducible values."""
    return RNG.randn(*shape).astype(np.float32)


@dataclass
class TinyModelSpec:
    """Tiny model dimensions used for fast plugin tests.

    All sizes are deliberately small so TRT engine builds complete in seconds.
    The head_dim is derived from hidden_size // num_attention_heads by default,
    but can be overridden for families with explicit head_dim configs.
    """

    vocab_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 4  # hidden_size // num_attention_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 128
    max_cache_length: int = 16


class FamilyPluginTester:
    """Base class for per-family plugin testers.

    Subclasses MUST set:
        plugin_module: str  -- importable module path (e.g. "tensorrt_model_connect.models.example")
        model_type: str     -- HF model_type string (e.g. "example_decoder")

    Subclasses MAY override:
        spec: TinyModelSpec -- custom dimensions for non-standard families
        get_config_dict()   -- custom config.json contents
        make_hf_tensors()   -- custom HF safetensors layout
        expected_weight_keys()       -- expected WeightDict keys after load_weights
        expected_engine_input_names()  -- expected TRT engine input tensor names
        expected_engine_output_names() -- expected TRT engine output tensor names
    """

    plugin_module: str = ""
    model_type: str = ""
    spec: TinyModelSpec = TinyModelSpec()

    def get_plugin(self) -> Any:
        """Import and return the family-owned model module under test.

        Skips the test with pytest.skip() if the module cannot be imported
        due to missing dependencies (e.g. tensorrt not installed).
        """
        try:
            mod = importlib.import_module(self.plugin_module)
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"Cannot import {self.plugin_module}: {exc}")
        return mod

    def get_config_dict(self) -> dict:
        """Return a minimal HF config.json dict for this family.

        The base implementation returns a standard decoder config compatible with
        decoder families that use the standard
        model_type + dimensions layout.

        Subclasses can override for families that require additional config keys
        (e.g. GPT-2 with ``n_embd`` instead of ``hidden_size``).
        """
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "hidden_size": s.hidden_size,
            "intermediate_size": s.intermediate_size,
            "num_hidden_layers": s.num_hidden_layers,
            "num_attention_heads": s.num_attention_heads,
            "num_key_value_heads": s.num_key_value_heads,
            "rms_norm_eps": s.rms_norm_eps,
            "rope_theta": s.rope_theta,
            "max_position_embeddings": s.max_position_embeddings,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Return synthetic weight tensors matching the HF safetensors layout.

        The base implementation creates the standard decoder weight layout used by
        standard decoder families:
          - model.embed_tokens.weight [vocab, hidden]
          - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
          - model.layers.{i}.mlp.{gate,up,down}_proj.weight
          - model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
          - model.norm.weight [hidden]
          - lm_head.weight [vocab, hidden]

        Subclasses should override for families with different HF weight key names
        (for example, fused-QKV or split-projection layouts).
        """
        s = self.spec
        kv_hidden = s.num_key_value_heads * s.head_dim

        # Reset RNG for deterministic output across calls.
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.embed_tokens.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.self_attn.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.o_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.mlp.gate_proj.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.up_proj.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.down_proj.weight"] = rand(s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def write_model_dir(self, tmp_path: Path) -> Path:
        """Write config.json and model.safetensors to tmp_path, return the dir.

        Creates a minimal synthetic model directory that can be loaded by
        ModelConfig.from_dir() and plugin.load_weights().
        """
        config_dict = self.get_config_dict()
        (tmp_path / "config.json").write_text(json.dumps(config_dict))

        tensors = self.make_hf_tensors()
        save_file(tensors, str(tmp_path / "model.safetensors"))
        return tmp_path

    def prepare_config_and_weights(
        self, tmp_path: Path,
    ) -> tuple[ModelConfig, WeightDict, dict[str, np.ndarray]]:
        """Create model dir, load config and weights, return all three.

        Returns:
            (config, weights, raw_hf_tensors) where:
            - config is the ModelConfig parsed from config.json
            - weights is the WeightDict returned by plugin.load_weights()
            - raw_hf_tensors is the dict of synthetic HF tensors before processing
        """
        raw_tensors = self.make_hf_tensors()
        config_dict = self.get_config_dict()
        (tmp_path / "config.json").write_text(json.dumps(config_dict))
        save_file(raw_tensors, str(tmp_path / "model.safetensors"))

        config = ModelConfig.from_dir(tmp_path)
        plugin = self.get_plugin()
        weights = plugin.load_weights(str(tmp_path), config)
        return config, weights, raw_tensors

    def expected_weight_keys(self) -> set[str]:
        """Return the set of weight keys the engine builder expects.

        The base implementation returns the standard decoder keys used by
        standard_decoder_builder.py:
          - embedding, final_norm, w_out
          - Per-layer: layer.{i}.w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
            input_norm, post_attn_norm

        Subclasses should override for families with additional or different keys
        (e.g. families with biases, per-head norms, position embeddings, etc.).
        """
        s = self.spec
        keys = {"embedding", "final_norm", "w_out"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.w_gate",
                f"{prefix}.w_up",
                f"{prefix}.w_down",
                f"{prefix}.input_norm",
                f"{prefix}.post_attn_norm",
            })
        return keys

    def expected_engine_input_names(self) -> set[str]:
        """Return expected TRT engine input tensor names.

        The base implementation returns the standard decoder inputs:
          token_id, position_id, attention_mask,
          cache_k_0 .. cache_k_{N-1}, cache_v_0 .. cache_v_{N-1}

        Subclasses should override for families with additional inputs
        (e.g. VL models with input_embed/use_input_embed).
        """
        s = self.spec
        names = {"token_id", "position_id", "attention_mask"}
        for i in range(s.num_hidden_layers):
            names.add(f"cache_k_{i}")
            names.add(f"cache_v_{i}")
        return names

    def expected_engine_output_names(self) -> set[str]:
        """Return expected TRT engine output tensor names.

        The base implementation returns the standard decoder outputs:
          logits, present_k_0 .. present_k_{N-1}, present_v_0 .. present_v_{N-1}

        Subclasses should override for families with additional outputs
        (e.g. hidden_state output for speech models).
        """
        s = self.spec
        names = {"logits"}
        for i in range(s.num_hidden_layers):
            names.add(f"present_k_{i}")
            names.add(f"present_v_{i}")
        return names
