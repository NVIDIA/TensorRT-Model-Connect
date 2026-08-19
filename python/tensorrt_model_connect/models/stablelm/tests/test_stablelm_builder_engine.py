# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for StableLM-2.

Intention:
    Validate the StableLM-2 family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    StableLM-2 uses LayerNorm (with beta) + SwiGLU MLP (gate/up/down) +
    RoPE + optional QKV biases. This test verifies LayerNorm betas and
    optional QKV biases are correctly mapped.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    make_hf_tensors() to produce StableLM's HF weight layout (LayerNorm with
    bias, SwiGLU MLP, QKV biases), and expected_weight_keys() to match the
    canonical keys with biases and norm betas.

Trace: ARCH-FAM-001, UD-FAM-STABLELM-01
Intent: Validate the StableLM-2 family plugin weight loading including LayerNorm with beta, SwiGLU MLP, QKV biases, and RoPE configuration.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: LayerNorm biases are loaded, QKV biases are mapped correctly, SwiGLU gate/up/down keys are present, and all weight shapes match expected dimensions.
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

from tensorrt_model_connect.models.stablelm.tests._family_plugin_tester import (
    FamilyPluginTester,
)
from tensorrt_model_connect.models.stablelm.tests._family_plugin_test_mixin import (
    FamilyPluginTestMixin,
)


class StableLMPluginTester(FamilyPluginTester):
    """Tester for the StableLM-2 family plugin.

    StableLM-2 uses:
      - LayerNorm (with beta) instead of RMSNorm
      - SwiGLU MLP (gate/up/down) -- same as LLaMA
      - RoPE for positional encoding
      - GQA (grouped query attention)
      - QKV biases
    """

    plugin_module = "tensorrt_model_connect.models.stablelm.model"
    model_type = "stablelm"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching StableLM's weight layout.

        Key differences from standard decoder:
          - LayerNorm with bias (input_layernorm.bias, post_attention_layernorm.bias)
          - SwiGLU MLP (gate_proj/up_proj/down_proj) -- standard naming
          - QKV biases
          - Final norm with bias
        """
        s = self.spec
        kv_hidden = s.num_key_value_heads * s.head_dim
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.embed_tokens.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            # LayerNorm with bias
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.input_layernorm.bias"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.bias"] = rand(s.hidden_size)
            # Q/K/V/O projections
            t[f"{p}.self_attn.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.o_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            # QKV biases
            t[f"{p}.self_attn.q_proj.bias"] = rand(s.hidden_size)
            t[f"{p}.self_attn.k_proj.bias"] = rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.bias"] = rand(kv_hidden)
            # SwiGLU MLP
            t[f"{p}.mlp.gate_proj.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.up_proj.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.down_proj.weight"] = rand(s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["model.norm.bias"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """StableLM weight keys: SwiGLU MLP + QKV biases + norm betas."""
        s = self.spec
        keys = {"embedding", "final_norm", "final_norm_beta", "w_out"}
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
                f"{prefix}.w_gate",
                f"{prefix}.w_up",
                f"{prefix}.w_down",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                f"{prefix}.post_attn_norm",
                f"{prefix}.post_attn_norm_beta",
            })
        return keys


class TestStableLMEngine(FamilyPluginTestMixin):
    """Engine tests for StableLM-2 family plugin."""

    tester_class = StableLMPluginTester
