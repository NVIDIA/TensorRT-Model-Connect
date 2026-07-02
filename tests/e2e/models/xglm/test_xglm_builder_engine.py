# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for XGLM.

Intention:
    Validate the XGLM family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    XGLM uses sinusoidal (computed) position embeddings, LayerNorm with beta,
    2-projection MLP (fc1/fc2), separate Q/K/V/O projections with biases,
    and non-standard config keys (d_model, ffn_dim, attention_heads, num_layers).
    The plugin also uses non-standard HF weight prefixes: self_attn_layer_norm,
    final_layer_norm, self_attn.out_proj, and top-level fc1/fc2.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    get_config_dict() for XGLM's config key names, make_hf_tensors() for
    XGLM's HF weight naming, and expected_weight_keys() for XGLM's canonical
    keys including position_embedding, fc1/fc2, QKV biases, output bias, and
    norm betas.

Trace: ARCH-FAM-001, UD-FAM-XGLM
Intent: Validate XGLM family plugin weight loading, key mapping, and engine build contract
Preconditions: Synthetic safetensors with XGLM-specific weight naming and config keys are available
Postconditions: Loaded WeightDict contains all expected canonical keys with correct shapes and transforms
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("safetensors not available", allow_module_level=True)

try:
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class XGLMPluginTester(FamilyPluginTester):
    """Tester for the XGLM family plugin.

    XGLM uses:
      - Sinusoidal position embeddings (computed, not learned)
      - LayerNorm (with beta)
      - 2-projection MLP (fc1/fc2) with GELU activation
      - Separate Q/K/V/O projections with biases
      - Config uses d_model, ffn_dim, attention_heads, num_layers
    """

    plugin_module = "tensorrt_model_connect.families.xglm"
    model_type = "xglm"

    def get_config_dict(self) -> dict:
        """XGLM uses non-standard config keys."""
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "d_model": s.hidden_size,
            "ffn_dim": s.intermediate_size,
            "attention_heads": s.num_attention_heads,
            "num_layers": s.num_hidden_layers,
            "max_position_embeddings": s.max_position_embeddings,
            "scale_embedding": False,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching XGLM's weight layout.

        Key differences from standard decoder:
          - self_attn_layer_norm (not input_layernorm)
          - final_layer_norm (not post_attention_layernorm)
          - self_attn.out_proj (not self_attn.o_proj)
          - Top-level fc1/fc2 (not mlp.gate_proj etc.)
          - All projections have biases
          - model.layer_norm (not model.norm) for final norm
        """
        s = self.spec
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.embed_tokens.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            # LayerNorm (pre-attn): self_attn_layer_norm
            t[f"{p}.self_attn_layer_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.self_attn_layer_norm.bias"] = rand(s.hidden_size)
            # LayerNorm (pre-MLP): final_layer_norm
            t[f"{p}.final_layer_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.final_layer_norm.bias"] = rand(s.hidden_size)
            # Q/K/V projections with biases
            t[f"{p}.self_attn.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.q_proj.bias"] = rand(s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.bias"] = rand(s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.v_proj.bias"] = rand(s.hidden_size)
            # Output projection: out_proj (not o_proj)
            t[f"{p}.self_attn.out_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.out_proj.bias"] = rand(s.hidden_size)
            # MLP: fc1/fc2 at top level
            t[f"{p}.fc1.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.fc1.bias"] = rand(s.intermediate_size)
            t[f"{p}.fc2.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.fc2.bias"] = rand(s.hidden_size)

        # Final LayerNorm: model.layer_norm (not model.norm)
        t["model.layer_norm.weight"] = rand(s.hidden_size)
        t["model.layer_norm.bias"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """XGLM weight keys: position_embedding + fc1/fc2 + biases + betas."""
        s = self.spec
        keys = {
            "embedding", "position_embedding",
            "final_norm", "final_norm_beta", "w_out",
        }
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.q_bias",
                f"{prefix}.k_bias",
                f"{prefix}.v_bias",
                f"{prefix}.o_bias",
                f"{prefix}.w_fc1",
                f"{prefix}.w_fc2",
                f"{prefix}.fc1_bias",
                f"{prefix}.fc2_bias",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                f"{prefix}.post_attn_norm",
                f"{prefix}.post_attn_norm_beta",
            })
        return keys

    def expected_engine_input_names(self) -> set[str]:
        """XGLM uses learned position type, which still uses standard inputs."""
        return super().expected_engine_input_names()


class TestXGLMEngine(FamilyPluginTestMixin):
    """Engine tests for XGLM family plugin."""

    tester_class = XGLMPluginTester
