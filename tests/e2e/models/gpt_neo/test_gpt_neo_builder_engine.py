# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for GPT-Neo.

Intention:
    Validate the GPT-Neo family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    GPT-Neo uses learned absolute position embeddings (wpe), LayerNorm with
    beta, separate Q/K/V Linear projections (under attn.attention.*), output
    projection with bias, GELU FC MLP (c_fc/c_proj as nn.Linear), tied word
    embeddings (wte == lm_head), and non-standard HF prefixes (transformer.h.*,
    transformer.wte, transformer.wpe).

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    make_hf_tensors() for GPT-Neo's unique HF weight layout, expected_weight_keys()
    for position_embedding + fc1/fc2 + biases + norm betas, and get_config_dict()
    for GPT-Neo config.

Trace: ARCH-FAM-001, UD-FAM-GPTNEO-01
Intent: Validate the GPT-Neo family plugin weight loading including learned positions, separate Q/K/V projections, GELU FC MLP, tied embeddings, and non-standard HF prefixes.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All weight keys map correctly from GPT-Neo's HF layout, position embeddings are loaded, biases are present, and engine IO tensors match expectations.
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


class GPTNeoPluginTester(FamilyPluginTester):
    """Tester for the GPT-Neo family plugin.

    GPT-Neo uses:
      - Learned absolute position embeddings (transformer.wpe)
      - LayerNorm (with beta)
      - Separate Q/K/V Linear projections (attn.attention.{q,k,v}_proj)
      - Output projection with bias (attn.attention.out_proj)
      - GELU FC MLP (c_fc/c_proj as nn.Linear)
      - Tied word embeddings (transformer.wte == lm_head)
    """

    plugin_module = "tensorrt_model_connect.families.gpt_neo"
    model_type = "gpt_neo"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching GPT-Neo's weight layout.

        Key differences:
          - transformer.wte.weight (token embedding)
          - transformer.wpe.weight (position embedding)
          - transformer.h.{i}.ln_1/ln_2 (LayerNorm naming)
          - transformer.h.{i}.attn.attention.{q,k,v}_proj.weight (separate Q/K/V)
          - transformer.h.{i}.attn.attention.out_proj.{weight,bias}
          - transformer.h.{i}.mlp.{c_fc,c_proj}.{weight,bias}
          - transformer.ln_f.{weight,bias} (final LayerNorm)
          - No lm_head.weight (tied to wte)
        """
        s = self.spec
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["transformer.wte.weight"] = rand(s.vocab_size, s.hidden_size)
        t["transformer.wpe.weight"] = rand(s.max_position_embeddings, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"transformer.h.{i}"
            # LayerNorm
            t[f"{p}.ln_1.weight"] = rand(s.hidden_size)
            t[f"{p}.ln_1.bias"] = rand(s.hidden_size)
            t[f"{p}.ln_2.weight"] = rand(s.hidden_size)
            t[f"{p}.ln_2.bias"] = rand(s.hidden_size)
            # Separate Q/K/V as Linear [out, in]
            t[f"{p}.attn.attention.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attn.attention.k_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attn.attention.v_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            # Output projection with bias
            t[f"{p}.attn.attention.out_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attn.attention.out_proj.bias"] = rand(s.hidden_size)
            # MLP: c_fc/c_proj as nn.Linear [out, in]
            t[f"{p}.mlp.c_fc.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.c_fc.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.c_proj.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.c_proj.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["transformer.ln_f.weight"] = rand(s.hidden_size)
        t["transformer.ln_f.bias"] = rand(s.hidden_size)
        # No lm_head -- tied to wte
        return t

    def expected_weight_keys(self) -> set[str]:
        """GPT-Neo weight keys: position_embedding + fc1/fc2 + biases + betas."""
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


class TestGPTNeoEngine(FamilyPluginTestMixin):
    """Engine tests for GPT-Neo family plugin."""

    tester_class = GPTNeoPluginTester
