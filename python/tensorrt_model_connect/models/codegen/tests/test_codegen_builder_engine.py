# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for CodeGen.

Intention:
    Validate the CodeGen family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    CodeGen uses fused QKV projection (qkv_proj) with mp_num=4 interleaving
    (Q, V, K order per chunk), parallel residual, single LayerNorm per block
    (ln_1 only, no ln_2), partial RoPE, GELU FC MLP (fc_in/fc_out), and
    non-standard HF prefixes (transformer.h.*, transformer.wte).

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    make_hf_tensors() for CodeGen's unique HF weight layout, expected_weight_keys()
    for fc1/fc2 MLP + biases (no post_attn_norm due to single-norm parallel
    residual), and get_config_dict() for CodeGen's rotary_dim config.

Trace: ARCH-FAM-001, UD-FAM-CODEGEN-01
Intent: Validate the CodeGen family plugin weight loading including fused QKV with mp_num interleaving, partial RoPE, parallel residual, and non-standard HF prefixes.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All weight keys are present with correct shapes, fused QKV is split correctly from interleaved layout, and engine IO tensors match expected names.
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

from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class CodeGenPluginTester(FamilyPluginTester):
    """Tester for the CodeGen family plugin.

    CodeGen uses:
      - LayerNorm (with beta)
      - Parallel residual connections
      - Fused QKV projection (qkv_proj) with mp_num=4 interleaving
      - Partial rotary embeddings (rotary_dim / head_dim)
      - Single LayerNorm per block (ln_1 only, no ln_2)
      - GELU FC MLP (fc_in/fc_out as Linear)
      - Separate lm_head with optional bias
    """

    plugin_module = "tensorrt_model_connect.models.codegen.model"
    model_type = "codegen"
    # CodeGen needs num_attention_heads divisible by mp_num=4
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
    )

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d["rotary_dim"] = 4  # head_dim = 4
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching CodeGen's weight layout.

        Key differences:
          - transformer.wte.weight (token embedding, no position embedding)
          - transformer.h.{i}.ln_1 (single LayerNorm per block)
          - transformer.h.{i}.attn.qkv_proj.weight [3*hidden, hidden]
            with mp_num=4 interleaving: Q, V, K order per chunk
          - transformer.h.{i}.attn.out_proj.weight (no bias)
          - transformer.h.{i}.mlp.fc_in/fc_out (Linear layout)
          - transformer.ln_f (final LayerNorm)
          - lm_head.weight (with optional lm_head.bias)
        """
        s = self.spec
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["transformer.wte.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"transformer.h.{i}"
            # Single LayerNorm
            t[f"{p}.ln_1.weight"] = rand(s.hidden_size)
            t[f"{p}.ln_1.bias"] = rand(s.hidden_size)
            # Fused QKV: [3*hidden, hidden]
            t[f"{p}.attn.qkv_proj.weight"] = rand(3 * s.hidden_size, s.hidden_size)
            # Output projection (no bias)
            t[f"{p}.attn.out_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            # MLP: fc_in/fc_out (Linear layout)
            t[f"{p}.mlp.fc_in.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.fc_in.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.fc_out.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.fc_out.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["transformer.ln_f.weight"] = rand(s.hidden_size)
        t["transformer.ln_f.bias"] = rand(s.hidden_size)
        # LM head (no bias in this test)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """CodeGen weight keys: fc1/fc2 + biases + single norm (no post_attn_norm)."""
        s = self.spec
        keys = {"embedding", "final_norm", "final_norm_beta", "w_out"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.w_fc1",
                f"{prefix}.w_fc2",
                f"{prefix}.fc1_bias",
                f"{prefix}.fc2_bias",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                # No post_attn_norm -- single-norm parallel residual
            })
        return keys


class TestCodeGenEngine(FamilyPluginTestMixin):
    """Engine tests for CodeGen family plugin."""

    tester_class = CodeGenPluginTester
