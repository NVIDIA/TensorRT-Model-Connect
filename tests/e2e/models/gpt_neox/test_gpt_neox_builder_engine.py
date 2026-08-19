# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for GPT-NeoX (Pythia).

Intention:
    Validate the GPT-NeoX family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    GPT-NeoX uses fused QKV projection (query_key_value), parallel residual,
    LayerNorm with beta, partial RoPE, GELU FC MLP (dense_h_to_4h/dense_4h_to_h),
    and non-standard HF key prefixes (gpt_neox.embed_in, gpt_neox.layers.*,
    embed_out). The fused QKV is interleaved per-head: for each head h,
    rows are [Q_h, K_h, V_h].

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    all of get_config_dict(), make_hf_tensors(), and expected_weight_keys()
    for GPT-NeoX's unique HF layout.

Trace: ARCH-FAM-001, UD-FAM-GPTNEOX-01
Intent: Validate the GPT-NeoX family plugin weight loading including per-head interleaved fused QKV splitting, parallel residual, partial RoPE, and non-standard HF prefixes.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Fused QKV is split from per-head interleaved layout, FC MLP keys resolve correctly, and all weight shapes match expected dimensions.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip("safetensors.numpy")
pytest.importorskip("tensorrt_model_connect.config")

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class GPTNeoXPluginTester(FamilyPluginTester):
    """Tester for the GPT-NeoX family plugin.

    GPT-NeoX / Pythia uses:
      - LayerNorm (with beta)
      - Parallel residual connections
      - Fused QKV projection (query_key_value) -- interleaved per head
      - Partial rotary embeddings (rotary_pct)
      - 2-projection MLP (dense_h_to_4h/dense_4h_to_h) with GELU
      - Non-standard HF prefixes: gpt_neox.embed_in, gpt_neox.layers.*
    """

    plugin_module = "tensorrt_model_connect.families.gpt_neox.model"
    model_type = "gpt_neox"

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d["rotary_pct"] = 0.5
        d["use_parallel_residual"] = True
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching GPT-NeoX's weight layout.

        Key differences:
          - gpt_neox.embed_in.weight (not model.embed_tokens.weight)
          - Fused QKV: gpt_neox.layers.{i}.attention.query_key_value.weight
            shape [3*hidden, hidden] -- interleaved per head
          - MLP: dense_h_to_4h/dense_4h_to_h
          - Final norm: gpt_neox.final_layer_norm
          - LM head: embed_out.weight
        """
        s = self.spec
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["gpt_neox.embed_in.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"gpt_neox.layers.{i}"
            # LayerNorms with bias
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.input_layernorm.bias"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.bias"] = rand(s.hidden_size)
            # Fused QKV: interleaved [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
            # Shape: [3*hidden, hidden]
            qkv = rand(3 * s.hidden_size, s.hidden_size)
            qkv_bias = rand(3 * s.hidden_size)
            t[f"{p}.attention.query_key_value.weight"] = qkv
            t[f"{p}.attention.query_key_value.bias"] = qkv_bias
            # Output projection
            t[f"{p}.attention.dense.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attention.dense.bias"] = rand(s.hidden_size)
            # MLP
            t[f"{p}.mlp.dense_h_to_4h.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["gpt_neox.final_layer_norm.weight"] = rand(s.hidden_size)
        t["gpt_neox.final_layer_norm.bias"] = rand(s.hidden_size)
        # LM head
        t["embed_out.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """GPT-NeoX weight keys: fc1/fc2 MLP + QKV/O biases + norm betas."""
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


class TestGPTNeoXEngine(FamilyPluginTestMixin):
    """Engine tests for GPT-NeoX family plugin."""

    tester_class = GPTNeoXPluginTester
