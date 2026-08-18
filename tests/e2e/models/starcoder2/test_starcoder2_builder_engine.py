# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for StarCoder2.

Intention:
    Validate the StarCoder2 family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    StarCoder2 uses LayerNorm (with beta) + GELU FC MLP (c_fc/c_proj) +
    RoPE + QKV biases + output projection bias. This test verifies the
    non-standard weight keys (fc1/fc2 instead of gate/up/down, norm betas,
    QKV biases, output bias) are all correctly mapped.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    make_hf_tensors() to produce StarCoder2's HF weight layout (LayerNorm
    with bias, c_fc/c_proj MLP naming, QKV biases, output bias), and
    expected_weight_keys() to match the canonical keys with biases and
    fc1/fc2 MLP keys.

Trace: ARCH-FAM-001, UD-FAM-STARCODER2-01
Intent: Validate the StarCoder2 family plugin weight loading including LayerNorm with beta, GELU FC MLP (c_fc/c_proj), QKV biases, and output projection bias.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: MLP keys map to fc1/fc2, all biases (QKV, output, norm) are loaded, and weight shapes match expected dimensions.
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


class StarCoder2PluginTester(FamilyPluginTester):
    """Tester for the StarCoder2 family plugin.

    StarCoder2 uses:
      - LayerNorm (with beta) instead of RMSNorm
      - GELU FC MLP (c_fc/c_proj) instead of SwiGLU (gate/up/down)
      - QKV biases + output projection bias
      - Separate Q/K/V/O projections with GQA
    """

    plugin_module = "tensorrt_model_connect.families.starcoder2.model"
    model_type = "starcoder2"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching StarCoder2's weight layout.

        Key differences from standard decoder:
          - LayerNorm with bias (input_layernorm.bias, post_attention_layernorm.bias)
          - MLP uses c_fc/c_proj naming (not gate/up/down)
          - QKV projections have biases
          - Output projection has bias
          - Final norm has bias
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
            # Output projection bias
            t[f"{p}.self_attn.o_proj.bias"] = rand(s.hidden_size)
            # MLP: c_fc/c_proj naming
            t[f"{p}.mlp.c_fc.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.c_fc.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.c_proj.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.c_proj.bias"] = rand(s.hidden_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["model.norm.bias"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """StarCoder2 weight keys: fc1/fc2 MLP + biases + norm betas."""
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


class TestStarCoder2Engine(FamilyPluginTestMixin):
    """Engine tests for StarCoder2 family plugin."""

    tester_class = StarCoder2PluginTester
